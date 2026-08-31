"""Phase 3b: compare three GOP formulas on the SAME frame-level posteriors.

Runs under the 3.14 torch env. Uses gop_data/ctc_head.pt (from train_gop_head.py).

Formulas (all from the head's frame-level posteriors z / p / logp):
  probability-GOP : GOP(p) = (1/|S|) * sum_t P(p | o_t)          (segment-avg posterior)
  logprob-GOP     : GOP(p) = (1/|S|) * sum_t log P(p | o_t)      (classic DNN-GOP/CCTC)
  max-logit GOP   : GOP(p) = max_{t in S} z_p(t)                  (segment-max logit)
  GOP-CTC-AF      : GOP(p) = min(L_perturbed) - L_original        (alignment-free, UPS)

Segments S come from reference-constrained CTC forced alignment (same frames for the
segment-based variants). Report Spearman + ROC-AUC at phone/word/sentence level.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

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
VARIANT_NAMES = ("prob", "logprob", "maxlogit", "ctcaf", "ctcaf_rps")
VARIANT_LABELS = {
    "prob": "probability-GOP",
    "logprob": "logprob-GOP",
    "maxlogit": "max-logit",
    "ctcaf": "GOP-CTC-AF(UPS)",
    "ctcaf_rps": "GOP-CTC-AF(RPS)",
}

# Restricted phoneme substitutions: articulatorily similar pairs + common
# Mandarin-L1 learner errors (single-phone substitutions only).
RPS_PAIRS = [
    ("P", "B"), ("T", "D"), ("K", "G"),
    ("F", "V"), ("TH", "S"), ("TH", "F"), ("TH", "T"), ("TH", "DH"),
    ("DH", "D"), ("DH", "Z"), ("DH", "V"),
    ("S", "Z"), ("SH", "S"), ("ZH", "Z"), ("CH", "SH"), ("CH", "JH"),
    ("JH", "Z"), ("JH", "D"),
    ("M", "N"), ("N", "NG"),
    ("L", "R"), ("R", "W"), ("L", "W"), ("V", "W"),
    ("IY", "IH"), ("EY", "EH"), ("EY", "IY"), ("AE", "EH"), ("AE", "AA"),
    ("AA", "AO"), ("AA", "AH"), ("AO", "OW"), ("OW", "UW"), ("UW", "UH"),
    ("AH", "ER"), ("ER", "R"), ("AY", "EY"), ("AW", "AO"), ("OY", "AA"),
    ("AE", "AH"),
]
RPS_CONFUSIONS: dict[str, set[str]] = {}
for _a, _b in RPS_PAIRS:
    RPS_CONFUSIONS.setdefault(_a, set()).add(_b)
    RPS_CONFUSIONS.setdefault(_b, set()).add(_a)


# ---------- shared: reference-constrained CTC forced alignment ----------
def ctc_forced_alignment(logp: np.ndarray, target: list[int], blank: int) -> list[list[int]]:
    """logp [T, C] log-probs -> per-target-phone frame lists (may be empty)."""
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


# ---------- three GOP formulas ----------
def segment_gop(probs: np.ndarray, logits: np.ndarray, target: list[int], blank: int):
    """probability-GOP, logprob-GOP, max-logit GOP from the same aligned segments."""
    logp = np.log(np.clip(probs, 1e-12, None))
    spans = ctc_forced_alignment(logp, target, blank)
    g_prob, g_logprob, g_max = [], [], []
    for i, fr in enumerate(spans):
        if not fr:
            g_prob.append(float("nan"))
            g_logprob.append(float("nan"))
            g_max.append(float("nan"))
        else:
            fr = np.asarray(fr)
            g_prob.append(float(np.mean(probs[fr, target[i]])))
            g_logprob.append(float(np.mean(logp[fr, target[i]])))
            g_max.append(float(np.max(logits[fr, target[i]])))
    return g_prob, g_logprob, g_max


def ctcaf_gop(
    logp_t: torch.Tensor,
    target: list[int],
    blank: int,
    loss_fn: nn.CTCLoss,
    rps: list[set[int]] | None = None,
) -> list[float]:
    """GOP-CTC-AF: min(L_perturbed) - L_original.

    rps=None -> UPS (substitute with every other phone).
    rps=[sets per phone id] -> RPS (restricted substitutions + deletion).
    """
    N = len(target)
    T, C = logp_t.shape
    tgt = torch.tensor(target, dtype=torch.long)
    orig = loss_fn(
        logp_t.unsqueeze(1), tgt.unsqueeze(0),
        torch.tensor([T]), torch.tensor([N]),
    ).item()

    seqs: list[list[int]] = []
    idx: list[int] = []
    for i in range(N):
        if rps is None:
            subs = [q for q in range(C - 1) if q != target[i]]
        else:
            subs = sorted(s for s in rps[target[i]] if s != target[i])
        for q in subs:
            seqs.append(target[:i] + [q] + target[i + 1:])
            idx.append(i)
        seqs.append(target[:i] + target[i + 1:])  # deletion
        idx.append(i)

    B = len(seqs)
    Nmax = max(len(s) for s in seqs)
    tgt_b = torch.full((B, Nmax), -1, dtype=torch.long)
    tlens = torch.zeros(B, dtype=torch.long)
    for j, s in enumerate(seqs):
        tgt_b[j, : len(s)] = torch.tensor(s)
        tlens[j] = len(s)
    logp_b = logp_t.unsqueeze(1).expand(T, B, C)
    losses = loss_fn(
        logp_b, tgt_b,
        torch.full((B,), T, dtype=torch.long), tlens,
    ).numpy()

    per_phone_min: dict[int, float] = {}
    for j, i in enumerate(idx):
        per_phone_min[i] = min(per_phone_min.get(i, float("inf")), float(losses[j]))
    return [per_phone_min[i] - orig for i in range(N)]


# ---------- metrics ----------
def spearman_auc(xs: list[float], accs: list[float]) -> tuple[float, float, int]:
    xs = np.asarray(xs, dtype=np.float64)
    accs = np.asarray(accs, dtype=np.float64)
    mask = np.isfinite(xs)
    if mask.sum() < 2:
        return float("nan"), float("nan"), 0
    rho, _ = spearmanr(xs[mask], accs[mask])
    y = (accs[mask] >= 2).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return rho, float("nan"), int(mask.sum())
    auc = roc_auc_score(y, xs[mask])  # higher GOP = more correct
    return rho, auc, int(mask.sum())


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=DATA_DIR / "ctc_head.pt")
    ap.add_argument("--upsample", type=int, default=1,
                    help="linear time-upsample factor (frame-rate probe)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    inv = json.loads((DATA_DIR / "inventory.json").read_text(encoding="utf-8"))
    split = json.loads((DATA_DIR / "split.json").read_text(encoding="utf-8"))
    V = len(inv)
    test_ids = split["test"]
    print(f"test utts: {len(test_ids)}  V: {V}")

    model = nn.Linear(1024, V + 1).to(device)
    model.load_state_dict(torch.load(args.head, map_location=device))
    model.eval()

    feats, ids = load_features(DATA_DIR / "feats")
    feat_map = {u: upsample_linear(normalize_feat(f), args.upsample) for u, f in zip(ids, feats)}
    loss_fn = nn.CTCLoss(blank=V, zero_infinity=True, reduction="none")
    scores = load_scores(DEFAULT_SCORES_PATH)
    rps_id_sets = [
        {inv.index(q) for q in RPS_CONFUSIONS.get(inv[i], set())} for i in range(V)
    ]

    phone_rows: dict[str, list] = {v: [] for v in VARIANT_NAMES}
    phone_rows["acc"] = []
    word_rows: dict[str, list] = {v: [] for v in VARIANT_NAMES}
    word_rows["acc"] = []
    sent_rows: dict[str, list] = {v: [] for v in VARIANT_NAMES}
    sent_rows["acc"], sent_rows["flu"] = [], []

    for utt_id in test_ids:
        if utt_id not in feat_map:
            continue
        with torch.no_grad():
            z = model(
                torch.from_numpy(feat_map[utt_id]).unsqueeze(1).to(device)
            ).squeeze(1).cpu()  # logits [T, C]
            logp = F.log_softmax(z, dim=-1)  # [T, C]
        p = torch.softmax(z, dim=-1).numpy()
        z = z.numpy()
        tgt = labels[utt_id]["phone_ids"]
        accs = labels[utt_id]["accs"]
        if len(tgt) != len(accs):
            continue

        g_prob, g_logprob, g_max = segment_gop(p, z, tgt, V)
        g_ctcaf = ctcaf_gop(logp, tgt, V, loss_fn)
        g_ctcaf_rps = ctcaf_gop(logp, tgt, V, loss_fn, rps_id_sets)
        g_all = {
            "prob": g_prob, "logprob": g_logprob, "maxlogit": g_max,
            "ctcaf": g_ctcaf, "ctcaf_rps": g_ctcaf_rps,
        }

        for v in VARIANT_NAMES:
            phone_rows[v].extend(g_all[v])
        phone_rows["acc"].extend(accs)

        # word-level via scores.json
        word_phones: list[list[str]] = []
        for w in utt_words(scores[utt_id]):
            word_phones.append([strip_stress(p) for p in w.get("phones", [])])
        flat = labels[utt_id]["phones"]
        i = 0
        for w_idx, wp in enumerate(word_phones):
            if w_idx >= len(scores[utt_id]["words"]):
                break
            n = len(wp)
            seg = slice(i, i + n)
            i += n
            if n == 0:
                continue
            wacc = scores[utt_id]["words"][w_idx].get("accuracy")
            if wacc is None:
                continue
            for v in VARIANT_NAMES:
                word_rows[v].append(
                    float(np.nanmean(np.asarray(g_all[v][seg], dtype=float)))
                )
            word_rows["acc"].append(float(wacc))

        for v in VARIANT_NAMES:
            sent_rows[v].append(float(np.nanmean(np.asarray(g_all[v], dtype=float))))
        sent_rows["acc"].append(float(scores[utt_id]["accuracy"]))
        sent_rows["flu"].append(float(scores[utt_id]["fluency"]))

    print("\n== phone-level (vs phones-accuracy 0-2) ==")
    for v in VARIANT_NAMES:
        rho, auc, n = spearman_auc(phone_rows[v], phone_rows["acc"])
        print(f"  {VARIANT_LABELS[v]:18s}: Spearman={rho:.3f}  AUC={auc:.3f}  n={n}")
    print("\n== word-level (vs word accuracy 0-10) ==")
    for v in VARIANT_NAMES:
        rho, auc, n = spearman_auc(word_rows[v], word_rows["acc"])
        print(f"  {VARIANT_LABELS[v]:18s}: Spearman={rho:.3f}  AUC={auc:.3f}  n={n}")
    print("\n== sentence-level ==")
    for v in VARIANT_NAMES:
        for target, tname in (("acc", "accuracy"), ("flu", "fluency")):
            rho, _, n = spearman_auc(sent_rows[v], sent_rows[target])
            print(f"  {VARIANT_LABELS[v]:18s} vs {tname:8s}: Spearman={rho:.3f}  n={n}")


if __name__ == "__main__":
    main()
