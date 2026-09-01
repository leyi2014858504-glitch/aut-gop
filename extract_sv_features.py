"""Extract SenseVoice-Small encoder features — second-encoder generalization probe.

Loads the frozen SenseVoiceSmall encoder (funasr, torch) and dumps encoder
outputs [T, D] per utt, reusing the SAME labels/inventory/split as the AuT
dataset dir. The CTC head dimension is auto-inferred downstream, so the whole
GOP recipe (train_gop_head -> gop_features -> train_gop_scorer) runs unchanged
with --data_dir.

Usage (3.14 env, after downloading iic/SenseVoiceSmall to models/SenseVoiceSmall):
  python extract_sv_features.py --out gop_data_sv                # SpeechOcean762
  python extract_sv_features.py --src gop_data_libri --out gop_data_sv_libri --from_manifest
Resumable: existing feats files are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import gop_data as gd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/SenseVoiceSmall")
    ap.add_argument("--src", default="gop_data",
                    help="dataset dir whose labels/feats to mirror")
    ap.add_argument("--out", default="gop_data_sv")
    ap.add_argument("--from_manifest", action="store_true",
                    help="wav paths from <src>/manifest.json (LibriSpeech mode)")
    ap.add_argument("--split", default=None,
                    help="manifest split to extract (implies --from_manifest), "
                         "e.g. test-clean -> feats_test/ + labels_test-clean.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    # auto-detect LibriSpeech-style dirs (SO762 ids break gop_data.utt_wav_path)
    from_manifest = args.from_manifest or (src / "manifest.json").is_file()

    if args.split:
        feats_sub, labels_dst = ("feats_test", f"labels_{args.split}.json")
    else:
        feats_sub, labels_dst = ("feats", "labels.json")
    (out / feats_sub).mkdir(parents=True, exist_ok=True)
    for fn in ("labels.json", "inventory.json", "split.json", "meta.json"):
        p = src / fn
        if p.exists() and not (out / fn).exists():
            shutil.copy(p, out / fn)

    if args.split:
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        labels = {u: v for u, v in manifest.items() if v["split"] == args.split}
        (out / labels_dst).write_text(
            json.dumps({u: {"phones": v["phones"], "phone_ids": v["phone_ids"],
                            "text": v["text"]} for u, v in labels.items()}),
            encoding="utf-8",
        )
    else:
        labels = json.loads((src / "labels.json").read_text(encoding="utf-8"))
    ids = sorted(labels.keys())
    if args.limit:
        ids = ids[: args.limit]

    import torch
    from funasr import AutoModel

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = AutoModel(model=args.model, disable_update=True, device=device)
    buf: dict = {}

    def _hook(_mod, _inp, output):
        buf["out"] = output[0] if isinstance(output, (tuple, list)) else output

    m.model.encoder.register_forward_hook(_hook)

    if from_manifest:
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        wav_of = lambda u: manifest[u]["wav"]  # noqa: E731
    else:
        wav_of = gd.utt_wav_path



    t0 = time.perf_counter()
    done = fail = 0
    for i, u in enumerate(ids):
        fp = out / feats_sub / f"{u}.npy"
        if fp.exists():
            continue
        try:
            m.generate(input=str(wav_of(u)), cache={}, language="en",
                       emotion="auto", event="auto", disable_pbar=True)
            x = buf["out"][0].detach().float().cpu().numpy()  # [T, D]
            if not np.isfinite(x).all():
                raise RuntimeError("non-finite encoder output")
            np.save(fp, x.astype(np.float16))
            if done == 0:
                print(f"sensevoice feats: [T,{x.shape[1]}] (frame rate = dur/{x.shape[0]})")
            done += 1
        except Exception as e:
            fail += 1
            print(f"  skip {u}: {e}")
            if fail > 20:
                raise SystemExit("too many failures — check model/API")
        if (i + 1) % 100 == 0:
            r = (i + 1) / (time.perf_counter() - t0)
            print(f"  [{i+1}/{len(ids)}] {r:.1f} utt/s done={done} fail={fail}")
    print(f"{out}: done={done} fail={fail} ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
