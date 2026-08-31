"""Leniency probe: does the head over-score non-native (SO762) vs native (LibriSpeech)?

Compares per-phone aligned mean log-posterior (logprob-GOP) distributions:
  - SO762 test, bucketed by human phones-accuracy (2 correct / 1 accented / 0 wrong)
  - LibriSpeech test-clean (native read speech, all phones assumed good)

If SO762 acc==2 confidence ≈ LibriSpeech confidence -> lenient (can't tell accent apart).
If SO762 acc==2 is clearly below native -> the model still discriminates accent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_gop import segment_gop
from gop_data import load_features, normalize_feat

DATA_DIR = Path("gop_data")
LIBRI_DIR = Path("gop_data_libri")


def collect_rows(model, feats_map, ids, targets, blank):
    rows = []
    with torch.no_grad():
        for u in ids:
            if u not in feats_map:
                continue
            z = model(
                torch.from_numpy(feats_map[u]).unsqueeze(1).to(device)
            ).squeeze(1).cpu()
            p = torch.softmax(z, dim=-1).numpy()
            _, loggop, _ = segment_gop(p, z.numpy(), targets[u], blank)
            rows.append((u, loggop))
    return rows


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    inv = json.loads((DATA_DIR / "inventory.json").read_text(encoding="utf-8"))
    split = json.loads((DATA_DIR / "split.json").read_text(encoding="utf-8"))
    V = len(inv)

    model = nn.Linear(1024, V + 1).to(device)
    model.load_state_dict(
        torch.load(DATA_DIR / "ctc_head_libri.pt", map_location=device)
    )
    model.eval()

    # --- SO762 test (non-native) ---
    feats, ids = load_features(DATA_DIR / "feats")
    feat_map = {u: normalize_feat(f) for u, f in zip(ids, feats)}
    so_rows = collect_rows(
        model, feat_map, split["test"],
        {u: labels[u]["phone_ids"] for u in split["test"]}, V,
    )
    del feats, feat_map

    # bucket SO762 phones by human score
    buckets = {2: [], 1: [], 0: []}
    for u, g in so_rows:
        accs = labels[u]["accs"]
        for v, a in zip(g, accs):
            buckets[int(a)].append(v)
    print("== SO762 test (non-native), phone-level mean log-posterior ==")
    for k in (2, 1, 0):
        a = np.asarray(buckets[k])
        print(f"  human={k} ({'correct/accented/wrong'[k]}): n={len(a)}  "
              f"mean={np.nanmean(a):.3f}  std={np.nanstd(a):.3f}")

    # --- LibriSpeech test-clean (native) ---
    lib_labels = json.loads(
        (LIBRI_DIR / "labels_test-clean.json").read_text(encoding="utf-8")
    )
    lfeats, lids = load_features(LIBRI_DIR / "feats_test")
    lmap = {u: normalize_feat(f) for u, f in zip(lids, lfeats)}
    lib_rows = collect_rows(
        model, lmap, lids, {u: lib_labels[u]["phone_ids"] for u in lids}, V,
    )
    del lfeats, lmap
    lib_all = np.concatenate([g for _, g in lib_rows])
    print(f"== LibriSpeech test-clean (native) ==")
    print(f"  all phones: n={len(lib_all)}  mean={np.nanmean(lib_all):.3f}  "
          f"std={np.nanstd(lib_all):.3f}")

    good = np.nanmean(np.asarray(buckets[2]))
    gap = good - float(np.nanmean(lib_all))
    print(f"\nleniency gap: SO762-correct mean - native mean = {gap:+.3f}")
    if gap > -0.2:
        print("=> lenient: accented-but-accepted phones score ~= native (over-scoring risk)")
    elif gap < -0.5:
        print("=> discriminating: model still separates accent from native")
    else:
        print("=> moderate gap")
