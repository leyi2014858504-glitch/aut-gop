"""Extract fused 25 Hz conv-tap features aligned to the existing 12.5 Hz feats.

Requires models/qwen3-asr-0.6b-int4/encoder.tap25.onnx (gelu_1 exposed as output).
For every utt already present in <dir>/feats/*.npy, run the tapped encoder on its
wav and save feats25/<utt>.npy = [T12, 960]: per 12.5 Hz frame, the two covered
25 Hz conv2 frames (freq-mean over the 32 freq bins) concatenated.

  python gop_extract_tap25.py --dirs gop_data gop_data_libri

Resumable: existing feats25 files are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import gop
from gop_data import load_wav_16k, utt_wav_path

TAP_MODEL = Path("models/qwen3-asr-0.6b-int4/encoder.tap25.onnx")
SRC_MODEL = Path("models/qwen3-asr-0.6b-int4/encoder.int4.onnx")
TOKENS_PER_WINDOW = 13
WIN25 = 25


def export_tap_onnx(src: Path = SRC_MODEL, dst: Path = TAP_MODEL, tap: str = "gelu_1") -> None:
    """Graph surgery: expose an intermediate tensor (25Hz conv tap) as a graph output.

    int4 quantization touches weights only; activations stay fp32, so tapping an
    intermediate activation is safe. Requires: pip install onnx.
    """
    import onnx

    m = onnx.load(str(src), load_external_data=False)
    if tap not in [o.name for o in m.graph.output]:
        m.graph.output.append(
            onnx.helper.make_tensor_value_info(tap, onnx.TensorProto.FLOAT, None)
        )
        onnx.save(m, str(dst))
        print(f"saved {dst} (graph output += {tap})")
    else:
        print(f"{src} already exposes {tap}")


def fuse25_to12(gelu1: np.ndarray, t12: int) -> np.ndarray:
    """[W,480,32,25] -> [t12,960]; token (i,p) covers in-window 25Hz frames 2p, 2p+1."""
    win = gelu1.transpose(0, 3, 1, 2).mean(axis=3)  # [W,25,480]
    w = win.shape[0]
    out = np.zeros((t12, 2 * win.shape[2]), dtype=np.float32)
    for j in range(t12):
        i, p = j // TOKENS_PER_WINDOW, j % TOKENS_PER_WINDOW
        i = min(i, w - 1)
        a = win[i, 2 * p]
        b = win[i, min(2 * p + 1, WIN25 - 1)]
        out[j] = np.concatenate([a, b])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["gop_data", "gop_data_libri"])
    ap.add_argument("--limit", type=int, default=0, help="max utts per dir (0 = all)")
    ap.add_argument("--export-tap-only", action="store_true",
                    help="just run graph surgery (encoder.int4.onnx -> encoder.tap25.onnx) and exit")
    args = ap.parse_args()

    if args.export_tap_only:
        export_tap_onnx()
        return

    sess = ort.InferenceSession(
        str(TAP_MODEL), providers=["CPUExecutionProvider"]
    )

    for d in map(Path, args.dirs):
        feats_dir = d / "feats"
        out_dir = d / "feats25"
        out_dir.mkdir(exist_ok=True)
        utts = sorted(p.stem for p in feats_dir.glob("*.npy"))
        if args.limit:
            utts = utts[: args.limit]

        # wav source: speechocean762 helper or LibriSpeech manifest
        if d == Path("gop_data"):
            wav_of = lambda u: utt_wav_path(u)  # noqa: E731
        else:
            manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            wav_of = lambda u: manifest[u]["wav"]  # noqa: E731

        t0 = time.perf_counter()
        done = fail = 0
        for k, u in enumerate(utts):
            if (out_dir / f"{u}.npy").exists():
                continue
            try:
                audio = load_wav_16k(wav_of(u))
                want_t = int(np.load(feats_dir / f"{u}.npy").shape[0])
                mel = gop._log_mel(audio)
                feat12, gelu1 = sess.run(["audio_features", "gelu_1"], {"mel": mel})
                if feat12.shape[1] != want_t:
                    raise RuntimeError(f"feats len {want_t} != encoder {feat12.shape[1]}")
                fused = fuse25_to12(gelu1, want_t)
                np.save(out_dir / f"{u}.npy", fused.astype(np.float16))
                done += 1
            except Exception as e:  # missing wav etc.
                fail += 1
                print(f"  skip {u}: {e}")
            if (k + 1) % 100 == 0:
                rate = (k + 1) / (time.perf_counter() - t0)
                print(f"  {d}: [{k+1}/{len(utts)}] {rate:.1f} utt/s, {done} done, {fail} fail")
        print(f"{d}: done={done} fail={fail} ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
