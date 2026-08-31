"""GOP data helpers: parse SpeechOcean762 scores.json, phone inventory, audio load.

Pure stdlib + numpy (runs under both the 3.11 inference env and the 3.14 torch env).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCORES_PATH = PROJECT_ROOT / "scores.json"
DEFAULT_WAV_ROOT = PROJECT_ROOT / "wavs" / "speechocean762-1.2.0" / "WAVE"

# mel hop of Qwen3-ASR (see asr.log_mel_spectrogram)
HOP_LEN = 160


def strip_stress(phone: str) -> str:
    """'IY0' -> 'IY' (drop ARPABET stress digits)."""
    return "".join(ch for ch in phone if not ch.isdigit())


def load_scores(path: Optional[Union[str, Path]] = None) -> dict:
    path = Path(path or DEFAULT_SCORES_PATH)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def utt_words(utt: dict) -> list[dict]:
    """words[] entries with 'phones' (list[str]) and 'phones-accuracy'."""
    return list(utt.get("words", []))


def utt_phone_list(utt: dict, strip: bool = True) -> list[str]:
    """Full utterance phone sequence, by concatenating words in order."""
    phones: list[str] = []
    for w in utt_words(utt):
        phones.extend(w.get("phones", []))
    return [strip_stress(p) if strip else p for p in phones]


def utt_phone_accs(utt: dict) -> list[float]:
    accs: list[float] = []
    for w in utt_words(utt):
        accs.extend(w.get("phones-accuracy", []))
    return accs


def build_phone_inventory(scores: dict) -> list[str]:
    """Sorted unique stripped phones across the corpus."""
    inv: set[str] = set()
    for utt in scores.values():
        inv.update(utt_phone_list(utt, strip=True))
    return sorted(inv)


def phone_to_id_map(inventory: list[str]) -> dict[str, int]:
    return {p: i for i, p in enumerate(inventory)}


def utt_phone_ids(utt: dict, inv: dict[str, int]) -> list[int]:
    return [inv[p] for p in utt_phone_list(utt, strip=True) if p in inv]


def utt_wav_path(utt_id: str, wav_root: Optional[Union[str, Path]] = None) -> Path:
    """'000010011' -> WAVE/SPEAKER0001/000010011.WAV."""
    root = Path(wav_root or DEFAULT_WAV_ROOT)
    speaker = int(utt_id[:5])
    return root / f"SPEAKER{speaker:04d}" / f"{utt_id}.WAV"


def list_local_utts(
    scores: dict,
    wav_root: Optional[Union[str, Path]] = None,
) -> list[str]:
    """utt ids present in scores AND having a local WAV file."""
    root = Path(wav_root or DEFAULT_WAV_ROOT)
    out = []
    for utt_id in scores:
        wav = utt_wav_path(utt_id, root)
        if wav.is_file():
            out.append(utt_id)
    return sorted(out)


def load_wav_16k(path: Union[str, Path]) -> np.ndarray:
    """Load WAV -> mono float32 16 kHz (soundfile + librosa fallback resample)."""
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(
            np.float32
        )
    return np.ascontiguousarray(audio, dtype=np.float32)


def mel_frame_count(audio_n: int) -> int:
    """librosa.stft center=True -> ~1 + n//hop frames (as asr.py uses)."""
    return max(1, 1 + audio_n // HOP_LEN)


# ---------- helpers shared by the torch (3.14) train/eval scripts ----------

def speaker_of(utt_id: str) -> int:
    """'000010011' -> 1 (SPEAKER0001)."""
    return int(utt_id[:5])


def load_features(data_dir: Union[str, Path]) -> tuple[list[np.ndarray], list[str]]:
    """Load gop_data/feats/*.npy -> (float32 arrays [T, 1024], utt_ids)."""
    data_dir = Path(data_dir)
    ids = sorted(p.stem for p in data_dir.glob("*.npy"))
    feats = [np.load(data_dir / f"{u}.npy").astype(np.float32) for u in ids]
    return feats, ids


def normalize_feat(f: np.ndarray) -> np.ndarray:
    """Per-utterance zero-mean / unit-var normalization."""
    f = f.astype(np.float32)
    m, s = f.mean(), f.std() + 1e-6
    return (f - m) / s


def upsample_linear(f: np.ndarray, k: int) -> np.ndarray:
    """Linear interpolate along time by factor k (frame-rate super-resolution probe)."""
    if k <= 1:
        return f
    T = f.shape[0]
    new_T = (T - 1) * k + 1
    idx = np.linspace(0, T - 1, new_T)
    i0 = idx.astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (idx - i0)[:, None]
    return (f[i0] * (1.0 - w) + f[i1] * w).astype(np.float32)


def speaker_split(
    utt_ids: list[str],
    seed: int = 42,
    train_frac: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Speaker-disjoint split (mirrors SpeechOcean762 official design)."""
    rng = np.random.default_rng(seed)
    spk = sorted({speaker_of(u) for u in utt_ids})
    rng.shuffle(spk)
    n_tr = int(round(len(spk) * train_frac))
    tr_spk, te_spk = set(spk[:n_tr]), set(spk[n_tr:])
    tr = [u for u in utt_ids if speaker_of(u) in tr_spk]
    te = [u for u in utt_ids if speaker_of(u) in te_spk]
    return tr, te


def load_utt_ids(path: Union[str, Path]) -> list[str]:
    """Load a plain-text utt-id list (one per line; kaldi 'text' first column)."""
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(line.split()[0])
    return out


def load_spk2utt_ids(path: Union[str, Path]) -> list[str]:
    """Load kaldi spk2utt: 'spk utt1 utt2 ...' -> flat utt id list."""
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out.extend(parts[1:])
    return out


def load_split_ids(path: Union[str, Path]) -> list[str]:
    """Auto-detect split format: kaldi spk2utt (multi-col) or flat utt list."""
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if lines and len(lines[0].split()) > 1:
        return load_spk2utt_ids(path)
    return load_utt_ids(path)
