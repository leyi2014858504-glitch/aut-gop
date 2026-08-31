"""
Qwen3-ASR ONNX inference (onnxruntime only, no PyTorch).

Inference flow follows andrewleech/qwen3-asr-onnx:
  audio -> log-mel -> encoder -> decoder_init (prefill) -> decoder_step loop
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnxruntime as ort

# Prompt special-token IDs (same across Qwen3-ASR sizes; also in config.json)
ENDOFTEXT_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
NEWLINE_TOKEN_ID = 198
EOS_TOKEN_IDS = {ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID}

CONV_WINDOW = 100
TOKENS_PER_WINDOW = 13


def _conv_out_len(t: int) -> int:
    return (t + 1) // 2


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    """Encoder token count from mel frame count (Qwen3-ASR formula)."""
    leave = input_lengths % CONV_WINDOW
    t = _conv_out_len(leave)
    t = _conv_out_len(t)
    t = _conv_out_len(t)
    return t + (input_lengths // CONV_WINDOW) * TOKENS_PER_WINDOW


def resolve_model_dir(model_dir: Optional[Union[str, Path]] = None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    root = Path(__file__).resolve().parent / "models"
    for candidate in (root, root / "qwen3-asr-0.6b-int4"):
        if (candidate / "decoder_init.int4.onnx").is_file() or (
            candidate / "encoder.int4.onnx"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "ONNX model dir not found. Expected files under ./models/ or "
        "./models/qwen3-asr-0.6b-int4/ (extract qwen3-asr-0.6b-int4.tar.gz)."
    )


def _find_file(model_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"None of {names} found in {model_dir}"
    )


def build_prompt_ids(audio_token_count: int) -> list[int]:
    """Build ASR chat-template token IDs with <|audio_pad|> x N."""
    ids = [
        IM_START_TOKEN_ID,
        9125,  # "system"
        NEWLINE_TOKEN_ID,
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID,
        882,  # "user"
        NEWLINE_TOKEN_ID,
        AUDIO_START_TOKEN_ID,
    ]
    ids.extend([AUDIO_PAD_TOKEN_ID] * audio_token_count)
    ids.extend(
        [
            AUDIO_END_TOKEN_ID,
            IM_END_TOKEN_ID,
            NEWLINE_TOKEN_ID,
            IM_START_TOKEN_ID,
            77091,  # "assistant"
            NEWLINE_TOKEN_ID,
        ]
    )
    return ids


def get_audio_pad_range(prompt_ids: list[int]) -> tuple[int, int]:
    start = end = None
    for i, tid in enumerate(prompt_ids):
        if tid == AUDIO_PAD_TOKEN_ID:
            if start is None:
                start = i
            end = i + 1
    if start is None or end is None:
        raise ValueError("No <|audio_pad|> tokens in prompt")
    return start, end


def log_mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Whisper-style log-mel spectrogram (numpy/librosa, no PyTorch).

    Mel params match config.json / Qwen3-ASR model card:
      16 kHz, 128 bins, n_fft=400 (25 ms), hop=160 (10 ms), Slaney, 0–8 kHz.

    Returns:
        float32 array shaped [1, n_mels, time]
    """
    import librosa

    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate}")
    x = np.asarray(audio, dtype=np.float32).reshape(-1)

    mel_filters = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        norm="slaney",
    ).astype(np.float32)

    stft = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    magnitudes = np.abs(stft) ** 2
    mel_spec = mel_filters @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    # Match WhisperFeatureExtractor / reference mel.py: drop last frame
    log_spec = log_spec[:, :-1]
    return log_spec[np.newaxis, :, :].astype(np.float32)


def _load_embed_tokens(model_dir: Path, config: dict) -> np.ndarray:
    dtype_name = config.get("embed_tokens_dtype", "float16")
    np_dtype = np.float16 if dtype_name == "float16" else np.float32
    vocab = int(config["decoder"]["vocab_size"])
    hidden = int(config["decoder"]["hidden_size"])
    shape = tuple(config.get("embed_tokens_shape", [vocab, hidden]))
    path = model_dir / "embed_tokens.bin"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name} — required for decoder_step embedding lookup"
        )
    embed = np.fromfile(path, dtype=np_dtype).reshape(shape)
    return embed.astype(np.float32, copy=False)


class Qwen3ASR:
    """Load encoder + decoder_init + decoder_step and transcribe 16 kHz audio."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        providers: Optional[list[str]] = None,
        max_new_tokens: int = 256,
    ):
        self.model_dir = resolve_model_dir(model_dir)
        self.max_new_tokens = max_new_tokens

        with open(self.model_dir / "config.json", encoding="utf-8") as f:
            self.config = json.load(f)

        self.mel_cfg = dict(self.config["mel"])
        special = self.config.get("special_tokens", {})
        self.eos_token_ids = set(special.get("eos_token_ids", list(EOS_TOKEN_IDS)))
        self.audio_pad_token_id = int(
            special.get("audio_pad_token_id", AUDIO_PAD_TOKEN_ID)
        )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        encoder_path = _find_file(
            self.model_dir, ("encoder.onnx", "encoder.int4.onnx")
        )
        decoder_init_path = _find_file(
            self.model_dir,
            ("decoder_init.int4.onnx", "decoder_init.onnx"),
        )
        decoder_step_path = _find_file(
            self.model_dir,
            ("decoder_step.int4.onnx", "decoder_step.onnx"),
        )

        self.encoder = ort.InferenceSession(
            str(encoder_path), sess_options=sess_opts, providers=providers
        )
        self.decoder_init = ort.InferenceSession(
            str(decoder_init_path), sess_options=sess_opts, providers=providers
        )
        self.decoder_step = ort.InferenceSession(
            str(decoder_step_path), sess_options=sess_opts, providers=providers
        )

        self.embed_tokens = _load_embed_tokens(self.model_dir, self.config)
        self._tokenizer = None

        init_inputs = {i.name for i in self.decoder_init.get_inputs()}
        self._init_uses_input_ids = "input_ids" in init_inputs

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            tok_path = self.model_dir / "tokenizer.json"
            if not tok_path.is_file():
                raise FileNotFoundError(f"Missing tokenizer: {tok_path}")
            self._tokenizer = Tokenizer.from_file(str(tok_path))
        return self._tokenizer

    def encode(self, audio_16k: np.ndarray) -> np.ndarray:
        """audio float32 16 kHz mono -> audio_features [1, T, 1024]."""
        mel = log_mel_spectrogram(audio_16k, **self.mel_cfg)
        (audio_features,) = self.encoder.run(
            ["audio_features"], {"mel": mel}
        )
        return audio_features

    def _greedy_decode(self, audio_features: np.ndarray) -> list[int]:
        audio_token_count = int(audio_features.shape[1])
        prompt_ids = build_prompt_ids(audio_token_count)
        # Rebuild pads with config token id if it differs (should not)
        if self.audio_pad_token_id != AUDIO_PAD_TOKEN_ID:
            prompt_ids = [
                self.audio_pad_token_id if t == AUDIO_PAD_TOKEN_ID else t
                for t in prompt_ids
            ]

        position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]

        if self._init_uses_input_ids:
            audio_start, _ = get_audio_pad_range(prompt_ids)
            input_ids = np.asarray(prompt_ids, dtype=np.int64)[np.newaxis, :]
            audio_offset = np.asarray([audio_start], dtype=np.int64)
            logits, present_keys, present_values = self.decoder_init.run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "audio_features": audio_features.astype(np.float32),
                    "audio_offset": audio_offset,
                },
            )
        else:
            # v1 format — kept for FP32 exports that take input_embeds
            input_embeds = self.embed_tokens[np.asarray(prompt_ids)].copy()
            audio_start, audio_end = get_audio_pad_range(prompt_ids)
            audio_len = audio_end - audio_start
            if audio_features.shape[1] != audio_len:
                raise ValueError(
                    f"audio_features length {audio_features.shape[1]} "
                    f"!= audio_pad count {audio_len}"
                )
            input_embeds[audio_start:audio_end] = audio_features[0]
            input_embeds = input_embeds[np.newaxis, :, :]
            logits, present_keys, present_values = self.decoder_init.run(
                ["logits", "present_keys", "present_values"],
                {"input_embeds": input_embeds, "position_ids": position_ids},
            )

        next_token = int(np.argmax(logits[0, -1, :]))
        output_tokens = [next_token]
        if next_token in self.eos_token_ids:
            return output_tokens

        pos = len(prompt_ids)
        for _ in range(self.max_new_tokens - 1):
            token_embed = self.embed_tokens[next_token][
                np.newaxis, np.newaxis, :
            ].astype(np.float32)
            step_pos = np.asarray([[pos]], dtype=np.int64)
            logits, present_keys, present_values = self.decoder_step.run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_embeds": token_embed,
                    "position_ids": step_pos,
                    "past_keys": present_keys,
                    "past_values": present_values,
                },
            )
            next_token = int(np.argmax(logits[0, -1, :]))
            output_tokens.append(next_token)
            pos += 1
            if next_token in self.eos_token_ids:
                break
        return output_tokens

    def transcribe(self, audio_16k: np.ndarray) -> str:
        """
        Transcribe 16 kHz mono float32 audio to text.

        Args:
            audio_16k: 1-D numpy float32, sample rate 16000
        """
        audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        audio_features = self.encode(audio)
        token_ids = self._greedy_decode(audio_features)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        # Qwen3-ASR emits: language <name><asr_text><transcript>
        marker = "<asr_text>"
        if marker in text:
            text = text.split(marker, 1)[1]
        return text.strip()

    def transcribe_with_timing(
        self, audio_16k: np.ndarray
    ) -> tuple[str, float, float]:
        """Return (text, wall_sec, audio_duration_sec)."""
        audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        audio_dur = float(len(audio)) / float(self.mel_cfg["sample_rate"])
        t0 = time.perf_counter()
        text = self.transcribe(audio)
        wall = time.perf_counter() - t0
        return text, wall, audio_dur
