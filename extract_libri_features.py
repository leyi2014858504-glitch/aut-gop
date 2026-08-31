"""Extract frozen AuT encoder features for LibriSpeech utts (GOP training/eval).

Runs under the 3.11 inference env. Input: gop_data_libri/manifest.json.
Output: gop_data_libri/feats/<utt>.npy (+ labels.json) for --split train
        gop_data_libri_test/... for --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from asr import Qwen3ASR
from gop_data import PROJECT_ROOT, load_wav_16k

DATA_DIR = PROJECT_ROOT / "gop_data_libri"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract LibriSpeech AuT features")
    ap.add_argument("--split", choices=["train-clean-100", "test-clean"], required=True)
    ap.add_argument("--max", type=int, default=0, help="0 = all in this split")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    utts = [u for u, v in manifest.items() if v["split"] == args.split]
    print(f"manifest {args.split}: {len(utts)} utts")

    rng = np.random.default_rng(args.seed)
    if args.max > 0 and args.max < len(utts):
        utts = list(rng.choice(utts, size=args.max, replace=False))
    print(f"extracting {len(utts)} utts")

    out_dir = DATA_DIR / ("feats" if args.split.startswith("train") else "feats_test")
    (DATA_DIR / out_dir.name).mkdir(parents=True, exist_ok=True)
    feats_dir = out_dir
    asr = Qwen3ASR()
    labels: dict[str, dict] = {}
    skipped = 0
    t0 = time.perf_counter()
    for i, utt in enumerate(utts):
        info = manifest[utt]
        try:
            audio = load_wav_16k(info["wav"])
            feats = asr.encode(audio)[0].astype(np.float16)  # [T, 1024]
        except Exception:
            skipped += 1
            continue
        np.save(feats_dir / f"{utt}.npy", feats)
        labels[utt] = {
            "phones": info["phones"],
            "phone_ids": info["phone_ids"],
            "text": info["text"],
        }
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(utts)}] ({time.perf_counter()-t0:.0f}s)")

    labels_name = "labels.json" if args.split.startswith("train") else "labels_test-clean.json"
    (feats_dir.parent / labels_name).write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )
    print(f"done: {len(labels)} utts, {skipped} failed, {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
