"""Temperature-scaling diagnostic for CTC head posteriors (Phase 11 follow-up).

Hypothesis under test: the DW-CNN head scores worse than the linear head
because its posteriors are over-sharpened (confident argmax -> saturated
logposteriors), not because its frame evidence is worse. If true, dividing
logits by a temperature T before log_softmax should recover GOP correlation.

Protocol:
  - T is fitted by minimizing CTC NLL on a held-out slice of the OFFICIAL
    TRAIN split only (never the test split).
  - We sweep a fixed grid to show correlation-vs-T (diagnostic upper bound).
  - Metrics mirror the official-table baselines exactly (raw, no XGBoost, no
    native calibration): phone AUC on per-phone logp_mean, word/sentence
    Spearman on mean logp.

Memory: stream one utt at a time, run the head forward ONCE, cache the tiny
[T, V+1] logits, then reuse them for every temperature. No whole-corpus copy.

Usage:
  py -3.14 -u probe_temperature.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import gop_features as gf
from gop_data import (
    DEFAULT_SCORES_PATH,
    load_scores,
    load_split_ids,
    normalize_feat,
    strip_stress,
)


@torch.no_grad()
def head_logits(model, feat: np.ndarray) -> np.ndarray:
    """[T, D] (already normalized) -> fp16 [T, V+1] logits on CPU."""
    z = model(torch.from_numpy(feat).unsqueeze(1)).squeeze(1)
    return z.float().numpy().astype(np.float16)


def nll_at(logit: np.ndarray, tgt: list[int], V: int, t: float, ctc) -> tuple[float, int]:
    lp = F.log_softmax(torch.from_numpy(logit.astype(np.float32)) / t, dim=-1)
    tg = torch.tensor(tgt, dtype=torch.int32)
    loss = ctc(lp.unsqueeze(1), tg.unsqueeze(0),
               torch.tensor([lp.shape[0]]), torch.tensor([len(tg)]))
    return float(loss), lp.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("gop_data"))
    ap.add_argument("--heads", nargs="+", default=["ctc_head_official.pt", "ctc_head_cnn_official.pt"])
    ap.add_argument("--grid", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--nll_ts", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    ap.add_argument("--nll_sub", type=int, default=600, help="train utts for NLL fit")
    ap.add_argument("--limit", type=int, default=0, help="cap eval utts (0 = all; for smoke)")
    args = ap.parse_args()

    inv = json.loads((args.data_dir / "inventory.json").read_text(encoding="utf-8"))
    V = len(inv)
    labels = json.loads((args.data_dir / "labels.json").read_text(encoding="utf-8"))
    scores = load_scores(DEFAULT_SCORES_PATH)
    ctc = torch.nn.CTCLoss(blank=V, reduction="none", zero_infinity=True)

    feats_dir = args.data_dir / "feats"

    def read_feat(u: str) -> np.ndarray:
        return normalize_feat(np.load(feats_dir / f"{u}.npy").astype(np.float32))

    tr_all = [u for u in load_split_ids("train_utt.txt")
              if (feats_dir / f"{u}.npy").is_file()]
    te_ids = [u for u in load_split_ids("test_utt.txt")
              if (feats_dir / f"{u}.npy").is_file()]
    if args.limit:
        te_ids = te_ids[:args.limit]
    random.seed(42)
    nll_ids = random.sample(tr_all, min(args.nll_sub, len(tr_all)))
    print(f"NLL fit: {len(nll_ids)} official-train | eval: {len(te_ids)} official-test utts", flush=True)

    for head in args.heads:
        model = gf.load_head(args.data_dir / head, V, "cpu")
        print(f"\n=== {head} ===", flush=True)

        # cache logits once (train subset for NLL + all test) — fp16, tiny
        print("  forward (train nll) ...", flush=True)
        tr_log = {u: head_logits(model, read_feat(u)) for u in nll_ids}
        nlls = []
        for t in args.nll_ts:
            tot = frames = 0
            for u, lg in tr_log.items():
                l, f = nll_at(lg, labels[u]["phone_ids"], V, t, ctc)
                tot += l
                frames += f
            nlls.append((t, tot / max(frames, 1)))
        best = min(nlls, key=lambda x: x[1])[0]
        print("  CTC-NLL/frame: " + "  ".join(f"T={t:g}:{v:.4f}" for t, v in nlls), flush=True)
        print(f"  NLL-optimal T* = {best:g}", flush=True)

        ts = sorted(set(args.grid + [best]))
        print("  forward (test) ...", flush=True)
        te_log = [(u, head_logits(model, read_feat(u))) for u in te_ids]

        # accumulate per-phone/word/sentence mean-logp + human targets per T
        acc = {t: {"p": [], "py": [], "w": [], "wy": [], "s": [], "sy": [], "sf": []} for t in ts}
        for u, lg in te_log:
            tgt = labels[u]["phone_ids"]
            accs = labels[u]["accs"]
            logit = lg.astype(np.float32)
            for t in ts:
                logp = F.log_softmax(torch.from_numpy(logit) / t, dim=-1).numpy()
                rows = gf.per_phone_rows(logp, tgt, V)
                if len(rows) != len(accs):
                    continue
                for r, a in zip(rows, accs):
                    if r is not None:
                        acc[t]["p"].append(r["logp_mean"])
                        acc[t]["py"].append(1.0 if a >= 2 else 0.0)
                if not any(r is None for r in rows):
                    acc[t]["s"].append(float(np.mean([r["logp_mean"] for r in rows])))
                    acc[t]["sy"].append(float(scores[u]["accuracy"]))
                    acc[t]["sf"].append(float(scores[u]["fluency"]))
                    i = 0
                    for w in scores[u]["words"]:
                        n = len([strip_stress(p) for p in w.get("phones", [])])
                        seg = rows[i:i + n]
                        i += n
                        wacc = w.get("accuracy")
                        if wacc is not None and n and all(r is not None for r in seg):
                            acc[t]["w"].append(float(np.mean([r["logp_mean"] for r in seg])))
                            acc[t]["wy"].append(float(wacc))

        print("   T   phoneAUC  wordRho  sentRho(acc) sentRho(flu)", flush=True)
        for t in ts:
            a = acc[t]
            try:
                auc = roc_auc_score(a["py"], a["p"])
            except ValueError:
                auc = float("nan")
            wr = spearmanr(a["w"], a["wy"]).statistic
            sr = spearmanr(a["s"], a["sy"]).statistic
            sf = spearmanr(a["s"], a["sf"]).statistic
            tag = " *" if t == best else ""
            print(f"  {t:4g}    {auc:.3f}     {wr:.3f}     {sr:.3f}      {sf:.3f}{tag}", flush=True)

    del tr_log, te_log
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
