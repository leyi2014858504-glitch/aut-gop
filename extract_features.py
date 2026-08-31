"""Phase 0: extract frozen AuT encoder features for a sample of SpeechOcean762.

Runs under the 3.11 inference env (onnxruntime / librosa).

Outputs (default ./gop_data/):
  feats/<utt_id>.npy   encoder features [T, 1024] float32
  labels.json          per-utt phones / phone_ids / accs
  inventory.json       stripped phone inventory (id -> phone)
  meta.json            sample params + skipped utts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from asr import Qwen3ASR
from gop_data import (
    DEFAULT_WAV_ROOT,
    build_phone_inventory,
    list_local_utts,
    load_scores,
    load_wav_16k,
    phone_to_id_map,
    utt_phone_accs,
    utt_phone_ids,
    utt_phone_list,
    utt_wav_path,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract AuT encoder features (GOP Phase 0)")
    ap.add_argument("--n_utts", type=int, default=0, help="0 = all local utts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("gop_data"))
    ap.add_argument(
        "--dtype", choices=["float16", "float32"], default="float16",
        help="feature storage dtype (float16 halves disk)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores()
    inventory = build_phone_inventory(scores)
    inv_map = phone_to_id_map(inventory)
    print(f"phone inventory: {len(inventory)} phones -> {inventory}")

    rng = np.random.default_rng(args.seed)
    all_utts = list_local_utts(scores)
    if args.n_utts > 0:
        rng.shuffle(all_utts)
        all_utts = all_utts[: args.n_utts]
    sampled = all_utts

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "feats").mkdir(exist_ok=True)

    asr = Qwen3ASR()
    labels: dict[str, dict] = {}
    skipped: list[str] = []
    t0 = time.perf_counter()
    for i, utt_id in enumerate(sampled):
        phones = utt_phone_list(scores[utt_id], strip=True)
        ids = utt_phone_ids(scores[utt_id], inv_map)
        # CTC cannot represent adjacent identical labels: skip such utts
        if any(a == b for a, b in zip(ids, ids[1:])):
            skipped.append(utt_id)
            continue
        audio = load_wav_16k(utt_wav_path(utt_id))
        feats = asr.encode(audio)[0].astype(args.dtype)  # [T, 1024]
        np.save(args.out / "feats" / f"{utt_id}.npy", feats)
        labels[utt_id] = {
            "phones": phones,
            "phone_ids": ids,
            "accs": utt_phone_accs(scores[utt_id]),
            "text": scores[utt_id]["text"],
        }
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(sampled)}] ({time.perf_counter()-t0:.0f}s)")

    (args.out / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (args.out / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "n_utts": len(sampled),
                "n_saved": len(labels),
                "skipped_adjacent_dup": skipped,
                "seed": args.seed,
                "enc_hz": 13.0,
                "dtype": args.dtype,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"done: {len(labels)} utts saved, {len(skipped)} skipped (adjacent dup phones), "
        f"{time.perf_counter()-t0:.0f}s"
    )


if __name__ == "__main__":
    main()
