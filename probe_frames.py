"""Phase 0 probe A: frames-per-phone statistics at the AuT encoder frame rate.

Runs under the 3.11 inference env (needs librosa/soundfile/onnxruntime).

Questions answered:
  1. How many encoder frames (12.5 Hz) fall per reference phone on average?
     (expect ~1-1.5; if much less, CCTC-style GOP will be noisy)
  2. Does the actual int4 encoder output length match the formula?
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np

from asr import Qwen3ASR, get_feat_extract_output_lengths
from gop_data import (
    DEFAULT_WAV_ROOT,
    list_local_utts,
    load_scores,
    load_wav_16k,
    mel_frame_count,
    utt_phone_list,
    utt_wav_path,
)

VERIFY_ENCODER_N = 5  # utts to run the real encoder on


def main() -> None:
    scores = load_scores()
    utts = list_local_utts(scores)
    print(f"scores: {len(scores)} utts, local WAVs: {len(utts)}")

    # ---- 1. per-phone encoder frames (formula-based, all local utts) ----
    frames_per_phone: list[float] = []
    mel_frames_list: list[int] = []
    for utt_id in utts:
        audio = load_wav_16k(utt_wav := utt_wav_path(utt_id))
        n_phones = len(utt_phone_list(scores[utt_id], strip=True))
        if n_phones == 0:
            continue
        m = mel_frame_count(len(audio))
        mel_frames_list.append(m)
        t = get_feat_extract_output_lengths(m)
        frames_per_phone.append(t / n_phones)

    fpp = np.asarray(frames_per_phone)
    print("\n== encoder frames per reference phone (n=%d utts) ==" % len(fpp))
    for name, fn in (
        ("mean", np.mean), ("median", np.median), ("p10", lambda a: np.percentile(a, 10)),
        ("p25", lambda a: np.percentile(a, 25)), ("p75", lambda a: np.percentile(a, 75)),
        ("p90", lambda a: np.percentile(a, 90)),
    ):
        print(f"  {name:<6}: {fn(fpp):.3f}")
    print(f"  utts with avg <1.0 frame/phone: {(fpp < 1.0).mean() * 100:.1f}%")
    print(f"  utts with avg <1.5 frame/phone: {(fpp < 1.5).mean() * 100:.1f}%")

    # ---- 2. verify real encoder output length vs formula ----
    print("\n== verify encoder output length (formula vs actual) ==")
    asr = Qwen3ASR()
    for utt_id in utts[:VERIFY_ENCODER_N]:
        audio = load_wav_16k(utt_wav := utt_wav_path(utt_id))
        m = mel_frame_count(len(audio))
        t_pred = get_feat_extract_output_lengths(m)
        feats = asr.encode(audio)
        t_actual = feats.shape[1]
        ok = "OK" if t_actual == t_pred else "MISMATCH"
        dur = len(audio) / 16000.0
        print(
            f"  {utt_id}: dur={dur:.2f}s mel={m} pred={t_pred} actual={t_actual} "
            f"({t_actual / dur:.1f} Hz, frame={1000 * dur / t_actual:.0f} ms) [{ok}]"
        )
        if t_actual != t_pred:
            raise SystemExit("FAIL: encoder length formula mismatch")


if __name__ == "__main__":
    main()
