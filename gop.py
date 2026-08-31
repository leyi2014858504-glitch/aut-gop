"""GOPScorer — standalone pronunciation scoring component.

Pipeline:
  audio (16 kHz float32) -> AuT encoder (ONNX) -> linear CTC head
  -> logprob-GOP features -> XGBoost scorers -> phone / word / sentence scores

Single-env runtime deps: onnxruntime, librosa, torch, xgboost,
phonemizer + espeakng-loader (G2P). All feature layouts match the trained
scorers (gop_data/scorer_*_cv_best_n200_d4_lr0.05.ubj, calibrated features).

Example:
    from gop import GOPScorer
    scorer = GOPScorer()
    result = scorer.score_audio(audio_16k, "WE CALL IT BEAR")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from g2p import text_to_arpabet_words

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "gop_data"
DEFAULT_ENCODER_DIR = PROJECT_ROOT / "models" / "qwen3-asr-0.6b-int4"
DEFAULT_HEAD = "ctc_head_libri.pt"
DEFAULT_SCORER_TAG = "cv_best_n200_d4_lr0.05"


# ---------- mel front-end (mirrors asr.py / Qwen3-ASR config) ----------
def _log_mel(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    import librosa

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    mel_filters = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
        norm="slaney",
    ).astype(np.float32)
    stft = librosa.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window="hann", center=True, pad_mode="reflect",
    )
    mel_spec = mel_filters @ (np.abs(stft) ** 2)
    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]  # match WhisperFeatureExtractor
    return log_spec[np.newaxis, :, :].astype(np.float32)


class _AuTEncoder:
    """AuT encoder ONNX (encoder.int4.onnx only — no decoder needed for GOP)."""

    def __init__(self, model_dir: Union[str, Path]):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.mel_cfg = dict(json.loads((model_dir / "config.json").read_text())["mel"])
        self.sess = ort.InferenceSession(
            str(model_dir / "encoder.int4.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, audio_16k: np.ndarray) -> np.ndarray:
        mel = _log_mel(audio_16k, **self.mel_cfg)
        (feats,) = self.sess.run(["audio_features"], {"mel": mel})
        return np.ascontiguousarray(feats[0], dtype=np.float32)  # [T, 1024]


# ---------- reference-constrained CTC forced alignment ----------
def _forced_alignment(
    logp: np.ndarray, target: list[int], blank: int
) -> list[list[int]]:
    T, C = logp.shape
    N = len(target)
    L = 2 * N + 1
    labels = np.empty(L, dtype=np.int64)
    labels[0] = blank
    for i in range(N):
        labels[2 * i + 1] = target[i]
        labels[2 * i + 2] = blank
    neg = -np.inf
    alpha = np.full((T, L), neg)
    alpha[0, 0] = logp[0, blank]
    if L > 1:
        alpha[0, 1] = logp[0, target[0]]
    for t in range(1, T):
        lrow = alpha[t - 1]
        for l in range(L):
            cur = lrow[l]
            if l >= 1:
                cur = np.logaddexp(cur, lrow[l - 1])
            if l >= 2 and labels[l] != labels[l - 2]:
                cur = np.logaddexp(cur, lrow[l - 2])
            alpha[t, l] = cur + logp[t, labels[l]]
    l = L - 2 if alpha[T - 1, L - 2] > alpha[T - 1, L - 1] else L - 1
    spans: list[list[int]] = [[] for _ in range(N)]
    for t in range(T - 1, -1, -1):
        if l % 2 == 1:
            spans[l // 2].append(t)
        if t == 0:
            break
        lrow = alpha[t - 1]
        best, bl = lrow[l], l
        if l >= 1 and lrow[l - 1] > best:
            best, bl = lrow[l - 1], l - 1
        if l >= 2 and labels[l] != labels[l - 2] and lrow[l - 2] > best:
            bl = l - 2
        l = bl
    return spans


def _per_phone_rows(logp: np.ndarray, target: list[int], blank: int):
    probs = np.exp(logp)
    spans = _forced_alignment(logp, target, blank)
    rows = []
    for i, fr in enumerate(spans):
        if not fr:
            rows.append(None)
            continue
        fr = np.asarray(fr)
        lp = logp[fr, target[i]]
        rows.append(
            {
                "logp_mean": float(lp.mean()),
                "logp_min": float(lp.min()),
                "logp_max": float(lp.max()),
                "logp_std": float(lp.std()),
                "prob_mean": float(probs[fr, target[i]].mean()),
                "dur": float(len(fr)),
                "phone": int(target[i]),
            }
        )
    return rows


def _agg(vals: list[float]) -> list[float]:
    a = np.asarray(vals, dtype=float)
    return [
        float(np.nanmean(a)), float(np.nanmin(a)), float(np.nanmax(a)),
        float(np.nanstd(a)),
        float(np.nanpercentile(a, 10)), float(np.nanpercentile(a, 90)),
    ]


class GOPScorer:
    """Score pronunciation quality of an utterance against its reference text."""

    def __init__(
        self,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        encoder_dir: Optional[Union[str, Path]] = DEFAULT_ENCODER_DIR,
        head: str = DEFAULT_HEAD,
        scorer_tag: str = DEFAULT_SCORER_TAG,
        device: Optional[str] = None,
    ):
        import torch
        import torch.nn as nn
        import xgboost as xgb

        data_dir = Path(data_dir)
        self.inv = json.loads((data_dir / "inventory.json").read_text(encoding="utf-8"))
        self.inv_map = {p: i for i, p in enumerate(self.inv)}
        self.V = len(self.inv)

        ns = np.load(data_dir / "native_stats.npz")
        self.native_mean = ns["mean"]
        self.native_std = ns["std"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.head = nn.Linear(1024, self.V + 1)
        self.head.load_state_dict(torch.load(data_dir / head, map_location=self.device))
        self.head.to(self.device).eval()

        self.scorer_phone = xgb.Booster()
        self.scorer_phone.load_model(str(data_dir / f"scorer_phone_{scorer_tag}.ubj"))
        self.scorer_word = xgb.Booster()
        self.scorer_word.load_model(str(data_dir / f"scorer_word_{scorer_tag}.ubj"))
        self.scorer_sent = xgb.Booster()
        self.scorer_sent.load_model(str(data_dir / f"scorer_sent_{scorer_tag}.ubj"))
        self.scorer_sentflu = xgb.Booster()
        self.scorer_sentflu.load_model(str(data_dir / f"scorer_sentflu_{scorer_tag}.ubj"))

        self.encoder = _AuTEncoder(encoder_dir) if encoder_dir else None

    # ---- public API ----
    def score_audio(self, audio_16k: np.ndarray, text: str) -> dict:
        """audio_16k: mono float32 @16 kHz; text: reference transcript."""
        if self.encoder is None:
            raise ValueError("encoder_dir was not provided")
        feats = self.encoder.encode(audio_16k)
        return self.score_features(feats, text)

    def score_features(self, feats: np.ndarray, text: str) -> dict:
        """feats: [T, 1024] AuT encoder features (frozen)."""
        import torch
        import torch.nn.functional as F

        # per-utterance normalize (matches training)
        f = feats.astype(np.float32)
        f = (f - f.mean()) / (f.std() + 1e-6)
        with torch.no_grad():
            logp = F.log_softmax(
                self.head(torch.from_numpy(f).unsqueeze(1).to(self.device)),
                dim=-1,
            ).squeeze(1).cpu().numpy()

        word_phones = text_to_arpabet_words(text)
        flat = [p for w in word_phones for p in w]
        ids = [self.inv_map[p] for p in flat]
        if len(ids) != len(flat):
            raise ValueError(f"unknown phones in text: {set(flat) - set(self.inv_map)}")
        rows = _per_phone_rows(logp, ids, self.V)

        def z(r: dict) -> float:
            m, s = self.native_mean[r["phone"]], self.native_std[r["phone"]]
            return (r["logp_mean"] - m) / s

        # ---- phone level ----
        phone_out = []
        for r in rows:
            if r is None:
                phone_out.append({"gop": None, "p_correct": None})
                continue
            feat = np.asarray(
                [r["logp_mean"], r["logp_min"], r["logp_max"], r["logp_std"],
                 r["prob_mean"], r["dur"], r["phone"], z(r)], dtype=np.float32
            ).reshape(1, -1)
            p = float(self.scorer_phone.predict(xgb_dmatrix(feat))[0])
            phone_out.append({"phone": self.inv[r["phone"]], "gop": r["logp_mean"],
                              "p_correct": p})

        # ---- word level (spans from per-word G2P) ----
        tokens = text.split()
        word_out = []
        i = 0
        for wi, wp in enumerate(word_phones):
            n = len(wp)
            seg = rows[i:i + n]
            i += n
            label = tokens[wi] if wi < len(tokens) else " ".join(wp)
            if not seg or any(r is None for r in seg):
                word_out.append({"word": label, "score": None})
                continue
            seg = [r for r in seg if r is not None]
            feat = np.asarray(
                _agg([r["logp_mean"] for r in seg])
                + _agg([z(r) for r in seg])
                + [len(seg), float(np.mean([r["dur"] for r in seg]))],
                dtype=np.float32,
            ).reshape(1, -1)
            word_out.append(
                {"word": label, "score": float(self.scorer_word.predict(xgb_dmatrix(feat))[0])}
            )

        # ---- sentence level ----
        sent = {}
        if rows and not any(r is None for r in rows):
            feat = np.asarray(
                _agg([r["logp_mean"] for r in rows])
                + _agg([z(r) for r in rows])
                + _agg([r["logp_min"] for r in rows])
                + [len(rows), float(np.mean([r["dur"] for r in rows]))],
                dtype=np.float32,
            ).reshape(1, -1)
            acc = float(self.scorer_sent.predict(xgb_dmatrix(feat))[0])
            flu = float(self.scorer_sentflu.predict(xgb_dmatrix(feat))[0])
            sent = {"accuracy": acc, "fluency": flu}

        return {"phones": phone_out, "words": word_out, "sentence": sent}


def xgb_dmatrix(X: np.ndarray):
    import xgboost as xgb

    return xgb.DMatrix(X)
