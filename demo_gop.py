"""End-to-end demo of the GOPScorer component (3.14 env).

Scores a few SpeechOcean762 test utterances and compares with human scores.
"""

from __future__ import annotations

import numpy as np

from gop import GOPScorer
from gop_data import (
    DEFAULT_SCORES_PATH,
    load_scores,
    load_wav_16k,
    utt_wav_path,
)

TEST_UTTS = ["000010011", "000010035", "001310168"]


def main() -> None:
    scores = load_scores(DEFAULT_SCORES_PATH)
    print("loading GOPScorer (encoder + head + scorers)...")
    scorer = GOPScorer()

    for utt_id in TEST_UTTS:
        if utt_id not in scores:
            print(f"  {utt_id}: not in scores, skip")
            continue
        wav = utt_wav_path(utt_id)
        if not wav.is_file():
            print(f"  {utt_id}: no wav, skip")
            continue
        text = scores[utt_id]["text"]
        audio = load_wav_16k(wav)
        res = scorer.score_audio(audio, text)
        human_acc = scores[utt_id]["accuracy"]
        human_flu = scores[utt_id]["fluency"]

        print(f"\n== {utt_id}: {text}")
        print(f"  human  accuracy={human_acc} fluency={human_flu}")
        sent = res["sentence"]
        print(f"  model  accuracy={sent.get('accuracy'):.1f} fluency={sent.get('fluency'):.1f}")
        phones = res["phones"]
        gops = [p["gop"] for p in phones if p["gop"] is not None]
        pcor = [p["p_correct"] for p in phones if p["p_correct"] is not None]
        print(f"  phones n={len(phones)} mean_gop={np.mean(gops):.2f} "
              f"mean_p_correct={np.mean(pcor):.2f}")
        ws = [w for w in res["words"] if w["score"] is not None]
        print(f"  words: " + ", ".join(f"{w['word']}={w['score']:.1f}" for w in ws[:5]))


if __name__ == "__main__":
    main()
