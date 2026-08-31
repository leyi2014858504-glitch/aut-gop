"""Evaluate greedy-decode phone error rate on LibriSpeech test-clean (3.14 torch).

Uses gop_data_libri/feats_test (extracted by extract_libri_features.py --split test-clean).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gop_data import load_features, normalize_feat

DATA_DIR = Path("gop_data_libri")


def greedy_decode(logp: np.ndarray, blank: int) -> list[int]:
    ids = logp.argmax(axis=1)
    out: list[int] = []
    prev = blank
    for t in ids:
        if t != blank and t != prev:
            out.append(int(t))
        prev = int(t)
    return out


def lev(a: list[int], b: list[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = json.loads((DATA_DIR / "labels_test-clean.json").read_text(encoding="utf-8"))
    inv = json.loads((DATA_DIR / "inventory.json").read_text(encoding="utf-8"))
    V = len(inv)

    head = args.head or (DATA_DIR.parent / "gop_data" / "ctc_head.pt")
    model = nn.Linear(1024, V + 1).to(device)
    model.load_state_dict(torch.load(head, map_location=device))
    model.eval()

    feats, ids = load_features(DATA_DIR / "feats_test")
    print(f"test-clean utts: {len(feats)}  V: {V}")
    errs = []
    with torch.no_grad():
        for i, (f, u) in enumerate(zip(feats, ids)):
            x = torch.from_numpy(normalize_feat(f)).unsqueeze(1).to(device)
            out = F.log_softmax(model(x), dim=-1).squeeze(1).cpu().numpy()
            hyp = greedy_decode(out, V)
            ref = labels[u]["phone_ids"]
            errs.append(lev(hyp, ref) / len(ref) if ref else 0.0)
    print(f"greedy PER on LibriSpeech test-clean: {np.mean(errs) * 100:.1f}%")


if __name__ == "__main__":
    main()
