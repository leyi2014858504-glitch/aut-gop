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

from gop_data import load_split_ids, normalize_feat, speaker_split, upsample_linear
from gop_heads import build_head

DATA_DIR = Path("gop_data")


def feat_paths(
    feats_dir: Path, utt: str, fuse25: bool, tap25_only: bool = False
) -> list[Path]:
    """Source npy files for one utt (feats/ plus feats25/ when fusing)."""
    if tap25_only:
        return [feats_dir.parent / "feats25" / f"{utt}.npy"]
    ps = [feats_dir / f"{utt}.npy"]
    if fuse25:
        ps.append(feats_dir.parent / "feats25" / f"{utt}.npy")
    return ps


def load_norm(paths: list[Path], upsample: int = 1) -> np.ndarray:
    """Lazily read + PART-WISE per-utt normalize + concat (RAM-safe, per batch only).

    Parts are normalized separately: feat12 and the gelu_1 tap differ ~5x in std,
    a single global normalization lets the high-variance part dominate the linear head.
    """
    arrs = [normalize_feat(np.load(p).astype(np.float32)) for p in paths]
    f = arrs[0] if len(arrs) == 1 else np.concatenate(arrs, axis=1)
    return upsample_linear(f, upsample)
EPOCHS = 20
BATCH = 32
LR = 1e-3
SEED = 42


def collate(
    feats: list[list[Path]], targets: list[list[int]], upsample: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lazily load batch utts, sort by length desc, pad; returns (feat[T,B,F], tlens, target, nlens)."""
    k = max(upsample, 1)
    lens = [
        (t if k <= 1 else (t - 1) * k + 1)
        for t in (np.load(ps[0], mmap_mode="r").shape[0] for ps in feats)
    ]
    order = sorted(range(len(feats)), key=lambda i: lens[i], reverse=True)
    Tmax = max(lens[i] for i in order)
    Nmax = max(len(targets[i]) for i in order)
    B = len(order)
    feat = None
    target = torch.full((B, Nmax), -1, dtype=torch.long)
    tlens, nlens = [], []
    for j, i in enumerate(order):
        f = torch.from_numpy(load_norm(feats[i], upsample))
        if feat is None:
            feat = torch.zeros(Tmax, B, f.shape[1])
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
    model: nn.Module, feat_paths: list[Path], ref: list[int], blank: int,
    upsample: int = 1,
) -> float:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(load_norm(feat_paths, upsample)).unsqueeze(1).to(device)
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
    ap.add_argument("--fuse25", action="store_true",
                    help="concat feats25/ (25Hz conv tap, window-aligned) -> 1984-dim input")
    ap.add_argument("--tap25_only", action="store_true",
                    help="use feats25/ alone (960-dim) — control for fusion experiment")
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR,
                    help="encoder dataset dir (gop_data / gop_data_sv ...)")
    ap.add_argument("--head_kind", choices=["linear", "cnn"], default="linear",
                    help="CTC head architecture (capacity ablation)")
    args = ap.parse_args()
    data_dir = args.data_dir

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else ''})")

    labels = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    inv = json.loads((data_dir / "inventory.json").read_text(encoding="utf-8"))
    V = len(inv)
    print(f"phone inventory: {V}  blank: {V}")

    ids = sorted(p.stem for p in (data_dir / "feats").glob("*.npy"))
    feat_set = set(ids)
    base_dim = int(np.load(data_dir / "feats" / f"{ids[0]}.npy", mmap_mode="r").shape[1])
    tap_dim = 0
    if args.fuse25 or args.tap25_only:
        n25 = len(list((data_dir / "feats25").glob("*.npy")))
        if n25 < len(ids):
            raise SystemExit(f"FAIL: feats25 incomplete ({n25}/{len(ids)})")
        tap_dim = int(np.load(data_dir / "feats25" / f"{ids[0]}.npy", mmap_mode="r").shape[1])
    dim0 = tap_dim if args.tap25_only else base_dim + tap_dim * int(args.fuse25)
    print(f"base feats: {len(ids)} utts ({dim0}-dim)")
    if args.train_ids_file and args.test_ids_file:
        tr_ids = [u for u in load_split_ids(args.train_ids_file) if u in feat_set]
        te_ids = [u for u in load_split_ids(args.test_ids_file) if u in feat_set]
        print(f"split override: train {len(tr_ids)} / test {len(te_ids)} (feature-filtered)")
    else:
        tr_ids, te_ids = speaker_split(ids, seed=SEED, train_frac=0.5)
        print(f"base split: train {len(tr_ids)} utts / test {len(te_ids)} utts")
        json.dump(
            {"train": tr_ids, "test": te_ids},
            open(data_dir / "split.json", "w"), indent=1,
        )

    def upaths(utt_dir: Path, u: str) -> list[Path]:
        return feat_paths(utt_dir, u, args.fuse25 or args.tap25_only, args.tap25_only)

    tr = [(upaths(data_dir / "feats", u), labels[u]["phone_ids"]) for u in tr_ids]
    te = [(upaths(data_dir / "feats", u), labels[u]["phone_ids"]) for u in te_ids]

    if args.extra is not None:
        extra_labels = json.loads(
            (args.extra / "labels.json").read_text(encoding="utf-8")
        )
        extra_ids = sorted(p.stem for p in (args.extra / "feats").glob("*.npy"))
        if not extra_ids:
            raise SystemExit(
                f"FAIL: --extra {args.extra}/feats is empty — refusing to silently "
                "train without the expansion set"
            )
        print(f"extra feats: {len(extra_ids)} utts (all -> train)")
        tr.extend((upaths(args.extra / "feats", u), extra_labels[u]["phone_ids"])
                  for u in extra_ids)

    print(f"final train: {len(tr)} utts / test: {len(te)} utts")
    din = dim0
    print(f"head: {args.head_kind}  input dim: {din}")
    model = build_head(args.head_kind, din, V + 1).to(device)
    print(f"head params: {sum(p.numel() for p in model.parameters()):,}")
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
            feat, tlens, target, nlens = collate(
                [tr[i][0] for i in idx], [tr[i][1] for i in idx], args.upsample
            )
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
            per_tr = np.mean([utt_per(model, f, t, V, args.upsample) for f, t in tr[:300]])
            per_te = np.mean([utt_per(model, f, t, V, args.upsample) for f, t in te[:300]])
            print(
                f"epoch {epoch:2d}  loss {total / n:.4f}  "
                f"PER(tr,300) {per_tr * 100:.1f}%  PER(te,300) {per_te * 100:.1f}%  "
                f"({time.perf_counter() - t0:.0f}s)"
            )

    per_tr = np.mean([utt_per(model, f, t, V, args.upsample) for f, t in tr])
    per_te = np.mean([utt_per(model, f, t, V, args.upsample) for f, t in te])
    print(f"\nfinal PER: train {per_tr * 100:.1f}%  test {per_te * 100:.1f}%")

    save_path = args.out or (data_dir / "ctc_head.pt")
    torch.save(model.state_dict(), save_path)
    print(f"saved: {save_path}")

    if per_te > 0.9:
        raise SystemExit("FAIL: test PER too high, head not learning")


if __name__ == "__main__":
    main()
