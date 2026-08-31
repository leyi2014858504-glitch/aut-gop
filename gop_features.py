"""GOP feature extraction for tree-based scorers (phone/word/sentence level).

All features derive from the frame-level posteriors of the (linear) CTC head.
Calibration: per-phone z-score vs NATIVE (LibriSpeech test-clean) statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_gop import ctc_forced_alignment
from gop_data import (
    DEFAULT_SCORES_PATH,
    load_features,
    load_scores,
    normalize_feat,
    strip_stress,
    upsample_linear,
    utt_words,
)

DATA_DIR = Path("gop_data")
LIBRI_DIR = Path("gop_data_libri")


def load_head(path: Path, V: int, device: str) -> nn.Module:
    model = nn.Linear(1024, V + 1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def utt_logp(model: nn.Module, feat: np.ndarray, device: str) -> np.ndarray:
    """Normalized feature -> log_softmax posteriors [T, C]."""
    with torch.no_grad():
        z = model(torch.from_numpy(feat).unsqueeze(1).to(device)).squeeze(1).cpu()
        return F.log_softmax(z, dim=-1).numpy()


def per_phone_rows(
    logp: np.ndarray, target: list[int], blank: int
) -> list[Optional[dict]]:
    """logp [T, C] -> per-phone feature dicts (None if phone has no aligned frame)."""
    probs = np.exp(logp)
    spans = ctc_forced_alignment(logp, target, blank)
    rows: list[Optional[dict]] = []
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


def native_phone_stats(model: nn.Module, V: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-phone mean/std of logp_mean over LibriSpeech test-clean (native reference)."""
    labels = json.loads((LIBRI_DIR / "labels_test-clean.json").read_text(encoding="utf-8"))
    feats, ids = load_features(LIBRI_DIR / "feats_test")
    sums = np.zeros(V)
    sums2 = np.zeros(V)
    cnt = np.zeros(V)
    for u, f in zip(ids, feats):
        logp = utt_logp(model, normalize_feat(f), device)
        for r in per_phone_rows(logp, labels[u]["phone_ids"], V):
            if r is not None:
                sums[r["phone"]] += r["logp_mean"]
                sums2[r["phone"]] += r["logp_mean"] ** 2
                cnt[r["phone"]] += 1
    mean = sums / np.maximum(cnt, 1)
    var = sums2 / np.maximum(cnt, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean, std


def agg_stats(vals: list[float]) -> list[float]:
    a = np.asarray(vals, dtype=float)
    return [
        float(np.nanmean(a)), float(np.nanmin(a)), float(np.nanmax(a)),
        float(np.nanstd(a)),
        float(np.nanpercentile(a, 10)), float(np.nanpercentile(a, 90)),
    ]


def build_datasets(model: nn.Module, V: int, device: str, use_calib: bool):
    """Extract phone/word/sentence feature matrices + targets for all SO762 utts.

    Returns dict of X/y arrays and the utt ids per row level.
    """
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    inv = json.loads((DATA_DIR / "inventory.json").read_text(encoding="utf-8"))
    split = json.loads((DATA_DIR / "split.json").read_text(encoding="utf-8"))
    scores = load_scores(DEFAULT_SCORES_PATH)

    native_mean, native_std = native_phone_stats(model, V, device)

    feats, ids = load_features(DATA_DIR / "feats")
    feat_map = {u: normalize_feat(f) for u, f in zip(ids, feats)}
    del feats

    def z(r: dict) -> float:
        m, s = native_mean[r["phone"]], native_std[r["phone"]]
        return (r["logp_mean"] - m) / s

    def phone_feat(r: dict) -> list[float]:
        f = [r["logp_mean"], r["logp_min"], r["logp_max"], r["logp_std"],
             r["prob_mean"], r["dur"], r["phone"]]
        if use_calib:
            f.append(z(r))
        return f

    P, W, S = [], [], []
    Py, Wy, Sy_acc, Sy_flu = [], [], [], []
    P_utt, W_utt, S_utt = [], [], []

    for u in ids:
        if u not in labels:
            continue
        logp = utt_logp(model, feat_map[u], device)
        tgt = labels[u]["phone_ids"]
        accs = labels[u]["accs"]
        rows = per_phone_rows(logp, tgt, V)
        if len(rows) != len(accs):
            continue

        # phone level
        for r, a in zip(rows, accs):
            if r is None:
                continue
            P.append(phone_feat(r))
            Py.append(1.0 if a >= 2 else 0.0)
            P_utt.append(u)

        # word level (via scores.json word phone spans)
        word_phones: list[list[str]] = [
            [strip_stress(p) for p in w.get("phones", [])] for w in utt_words(scores[u])
        ]
        i = 0
        for w_idx, wp in enumerate(word_phones):
            if w_idx >= len(scores[u]["words"]):
                break
            n = len(wp)
            seg = rows[i:i + n]
            i += n
            wacc = scores[u]["words"][w_idx].get("accuracy")
            if wacc is None or any(r is None for r in seg):
                continue
            seg = [r for r in seg if r is not None]
            if not seg:
                continue
            W.append(
                agg_stats([r["logp_mean"] for r in seg])
                + agg_stats([z(r) for r in seg])
                + [len(seg), np.mean([r["dur"] for r in seg])]
            )
            Wy.append(float(wacc))
            W_utt.append(u)

        # sentence level
        if not any(r is None for r in rows):
            S.append(
                agg_stats([r["logp_mean"] for r in rows])
                + agg_stats([z(r) for r in rows])
                + agg_stats([r["logp_min"] for r in rows])
                + [len(rows), np.mean([r["dur"] for r in rows])]
            )
            Sy_acc.append(float(scores[u]["accuracy"]))
            Sy_flu.append(float(scores[u]["fluency"]))
            S_utt.append(u)

    del feat_map
    return {
        "phone": (np.asarray(P, dtype=np.float32), np.asarray(Py), P_utt),
        "word": (np.asarray(W, dtype=np.float32), np.asarray(Wy), W_utt),
        "sentence": (np.asarray(S, dtype=np.float32), Sy_acc, Sy_flu, S_utt),
    }, split, inv
