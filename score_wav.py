"""Score a WAV against a reference text with the GOPScorer component.

Usage (3.14 env):
    py -3.14 score_wav.py --wav my.wav --text "WE CALL IT BEAR"

The WAV may be any sample rate / channels (auto-resampled to 16k mono).
"""

from __future__ import annotations

import argparse

from gop import GOPScorer
from gop_data import load_wav_16k


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a WAV against reference text")
    ap.add_argument("--wav", required=True, help="path to audio file (wav/flac/mp3...)")
    ap.add_argument("--text", required=True, help="reference transcript, e.g. 'WE CALL IT BEAR'")
    ap.add_argument("--head", default="ctc_head_libri.pt")
    ap.add_argument("--scorer_tag", default="cv_best_n200_d4_lr0.05")
    args = ap.parse_args()

    scorer = GOPScorer(head=args.head, scorer_tag=args.scorer_tag)
    audio = load_wav_16k(args.wav)
    print(f"audio: {args.wav}  ({len(audio) / 16000.0:.2f}s @16k)  text: {args.text}")

    res = scorer.score_audio(audio, args.text)
    sent = res["sentence"]
    print(f"\nsentence accuracy: {sent.get('accuracy', float('nan')):.1f}  "
          f"fluency: {sent.get('fluency', float('nan')):.1f}")

    print("\nwords:")
    for w in res["words"]:
        s = w["score"]
        print(f"  {w['word']:20s} {s:5.1f}" if s is not None else f"  {w['word']:20s}    -")

    print("\nphones (top suspicious first):")
    phones = [p for p in res["phones"] if p["gop"] is not None]
    for p in sorted(phones, key=lambda x: x["p_correct"] if x["p_correct"] is not None else 1.0)[:8]:
        print(f"  {p['phone']:4s} gop={p['gop']:7.2f}  p_correct={p['p_correct']:.2f}")


if __name__ == "__main__":
    main()
