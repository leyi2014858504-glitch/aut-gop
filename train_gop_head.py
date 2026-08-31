"""Phase 2: train the linear CTC head on frozen AuT encoder features.

Runs under the 3.14 torch env (python -3.14), CUDA if available.

Split: speaker-disjoint (mirrors SpeechOcean762 official design).
Output: gop_data/ctc_head.pt (state_dict) + gop_data/split.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gop_data import load_features, load_split_ids, normalize_feat, speaker_split, upsample_linear

DATA_DIR = Path("gop_data")
EPOCHS = 20
BATCH = 32
LR = 1e-3
SEED = 42


def collate(
    feats: list[np.ndarray], targets: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort by length desc, pad features+targets; returns (feats[T,B,F], tlens, targets, tlens_t)."""
    order = sorted(range(len(feats)), key=lambda i: feats[i].shape[0], reverse=True)
    Tmax = max(feats[i].shape[0] for i in order)
    Nmax = max(len(targets[i]) for i in order)
    B = len(order)
    feat = torch.zeros(Tmax, B, feats[order[0]].shape[1])
    target = torch.full((B, Nmax), -1, dtype=torch.long)
    tlens, nlens = [], []
    for j, i in enumerate(order):
        f = torch.from_numpy(feats[i])
        feat[: f.shape[0], j] = f
        tlens.append(f.shape[0])
        t = torch.tensor(targets[i], dtype=torch.long)
        target[j, : len(t)] = t
        nlens.append(len(t))
    return feat, torch.tensor(tlens), target, torch.tensor(nlens)


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


def utt_per(
    model: nn.Module, feat: np.ndarray, ref: list[int], blank: int
) -> float:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feat).unsqueeze(1).to(device)
        out = F.log_softmax(model(x), dim=-1).squeeze(1).cpu().numpy()
    hyp = greedy_decode(out, blank)
    return lev(hyp, ref) / len(ref)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=Path, default=None,
                    help="extra feature dir (feats/ + labels.json), all go to train")
    ap.add_argument("--out", type=Path, default=None,
                    help="head save path (default gop_data/ctc_head.pt)")
    ap.add_argument("--upsample", type=int, default=1,
                    help="linear time-upsample factor for frame-rate probe")
    ap.add_argument("--train_ids_file", type=Path, default=None,
                    help="override train utt ids (one per line)")
    ap.add_argument("--test_ids_file", type=Path, default=None,
                    help="override test utt ids (one per line)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else ''})")

    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    inv = json.loads((DATA_DIR / "inventory.json").read_text(encoding="utf-8"))
    V = len(inv)
    print(f"phone inventory: {V}  blank: {V}")

    feats, ids = load_features(DATA_DIR / "feats")
    feat_set = set(ids)
    print(f"base feats: {len(feats)} utts")
    if args.train_ids_file and args.test_ids_file:
        tr_ids = [u for u in load_split_ids(args.train_ids_file) if u in feat_set]
        te_ids = [u for u in load_split_ids(args.test_ids_file) if u in feat_set]
        print(f"split override: train {len(tr_ids)} / test {len(te_ids)} (feature-filtered)")
    else:
        tr_ids, te_ids = speaker_split(ids, seed=SEED, train_frac=0.5)
        print(f"base split: train {len(tr_ids)} utts / test {len(te_ids)} utts")
        json.dump(
            {"train": tr_ids, "test": te_ids},
            open(DATA_DIR / "split.json", "w"), indent=1,
        )

    feat_map = {u: upsample_linear(normalize_feat(f), args.upsample) for u, f in zip(ids, feats)}
    tr = [(feat_map[u], labels[u]["phone_ids"]) for u in tr_ids]
    te = [(feat_map[u], labels[u]["phone_ids"]) for u in te_ids]
    del feats, feat_map

    if args.extra is not None:
        extra_labels = json.loads(
            (args.extra / "labels.json").read_text(encoding="utf-8")
        )
        extra_feats, extra_ids = load_features(args.extra / "feats")
        print(f"extra feats: {len(extra_feats)} utts (all -> train)")
        for u, f in zip(extra_ids, extra_feats):
            tr.append((upsample_linear(normalize_feat(f), args.upsample), extra_labels[u]["phone_ids"]))
        del extra_feats

    print(f"final train: {len(tr)} utts / test: {len(te)} utts")
    model = nn.Linear(1024, V + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CTCLoss(blank=V, zero_infinity=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        perm = rng.permutation(len(tr))
        total, n = 0.0, 0
        for b0 in range(0, len(tr), BATCH):
            idx = [perm[i] for i in range(b0, min(b0 + BATCH, len(tr)))]
            feat, tlens, target, nlens = collate([tr[i][0] for i in idx], [tr[i][1] for i in idx])
            feat, target = feat.to(device), target.to(device)
            logp = F.log_softmax(model(feat), dim=-1)
            loss = loss_fn(logp, target, tlens, nlens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        sched.step()
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            per_tr = np.mean([utt_per(model, f, t, V) for f, t in tr[:300]])
            per_te = np.mean([utt_per(model, f, t, V) for f, t in te[:300]])
            print(
                f"epoch {epoch:2d}  loss {total / n:.4f}  "
                f"PER(tr,300) {per_tr * 100:.1f}%  PER(te,300) {per_te * 100:.1f}%  "
                f"({time.perf_counter() - t0:.0f}s)"
            )

    per_tr = np.mean([utt_per(model, f, t, V) for f, t in tr])
    per_te = np.mean([utt_per(model, f, t, V) for f, t in te])
    print(f"\nfinal PER: train {per_tr * 100:.1f}%  test {per_te * 100:.1f}%")

    save_path = args.out or (DATA_DIR / "ctc_head.pt")
    torch.save(model.state_dict(), save_path)
    print(f"saved: {save_path}")

    if per_te > 0.9:
        raise SystemExit("FAIL: test PER too high, head not learning")


if __name__ == "__main__":
    main()
