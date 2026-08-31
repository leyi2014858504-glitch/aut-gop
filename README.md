# AuT-GOP

**Goodness of Pronunciation (GOP) scoring from a frozen lightweight ASR encoder** — no wav2vec2, no fine-tuned LLM, web-deployable.

Pipeline: `audio (16 kHz) -> frozen AuT encoder (Qwen3-ASR-0.6B, ONNX int4) -> single linear CTC head -> reference-constrained Viterbi alignment -> logprob-GOP -> gradient-boosted aggregation (XGBoost) -> phone / word / sentence scores`.

Trainable part is tiny: one linear CTC head (162 KB) + 4 XGBoost scorers (~4 MB). Everything heavier is frozen and shared with the ASR you may already run.

## Results (SpeechOcean762, official 2500/2500 speaker-disjoint split)

Head retrained on the official train split only + LibriSpeech (no test-speaker leakage).

| Level | Baseline (mean logprob-GOP) | AuT-GOP (XGBoost) |
|---|---|---|
| Phone AUC (correct/incorrect) | 0.723 | **0.815** |
| Word Spearman / PCC | 0.395 | 0.390 / **0.611** |
| Sentence accuracy Spearman / PCC | 0.426 | **0.553 / 0.682** |
| Sentence fluency Spearman / PCC | 0.279 | **0.529 / 0.625** |

- CTC head PER on the official test split: 28.0 % (24.7 % on train).
- 5-fold speaker-disjoint CV (in-domain split, robustness): sentence 0.625 ± 0.047, phone AUC 0.825 ± 0.008.
- Sentence-level PCC 0.682 is on par with published fine-tuned systems on SpeechOcean762 (E2E-R ≈ 0.68, GOPT ≈ 0.68; PCC protocol).

## Quick start

Requirements: Python 3.10+ (developed on 3.14), CPU inference works (`onnxruntime`).

```bash
pip install -r requirements.txt
```

1. Download the frozen encoder from [Qwen3-ASR-0.6B-int4](https://huggingface.co/) and place **only** `config.json` + `encoder.int4.onnx` (~711 MB) under `models/qwen3-asr-0.6b-int4/`. (The decoder is never loaded for GOP.)
2. Score any wav against a reference transcript:

```bash
python score_wav.py --wav my.wav --text "WE CALL IT BEAR" \
    --head ctc_head_official.pt --scorer_tag official
```

Output: sentence accuracy / fluency (0–10), per-word scores, and the most suspicious phones with per-phone `p_correct`.

Library use:

```python
from gop import GOPScorer
from gop_data import load_wav_16k

scorer = GOPScorer(head="ctc_head_official.pt", scorer_tag="official")
result = scorer.score_audio(load_wav_16k("my.wav"), "WE CALL IT BEAR")
# result: {"phones": [...], "words": [...], "sentence": {"accuracy": .., "fluency": ..}}
```

## Repository layout

| File | Role |
|---|---|
| `gop.py` | `GOPScorer` deployable component (encoder ONNX + CTC head + alignment + scorers) |
| `gop_data.py` | data helpers: wav loading, splits (kaldi spk2utt or flat), scores |
| `g2p.py` | text -> ARPABET-39 via phonemizer/espeak-ng (IPA bridge) |
| `eval_gop.py` | CTC forced alignment + GOP variants (prob / logprob / max-logit / AF) |
| `extract_features.py` | dump frozen-encoder features `[T,1024]` for SpeechOcean762 |
| `train_gop_head.py` | train the single linear CTC head (CTC loss) |
| `gop_features.py` | build phone/word/sentence GOP feature matrices (+ native-z calibration) |
| `train_gop_scorer.py` | train XGBoost scorers (official or ad-hoc split) |
| `cv_gop_scorer.py` | 5-fold speaker-disjoint CV + hyper-parameter grid |
| `probe_leniency.py` | native vs non-native leniency probe |
| `probe_frames.py` | frame-rate (upsampling) ablation |
| `build_libri_manifest.py`, `extract_libri_features.py`, `eval_libri_per.py` | LibriSpeech cross-domain expansion |
| `gop_data/` | released weights: CTC heads (`ctc_head_official.pt`, `ctc_head_libri.pt`), XGBoost scorers (`official`, `cv_best_n200_d4_lr0.05` tags), `inventory.json`, `labels.json`, `native_stats.npz` |

## Reproducing the training pipeline

Datasets are **not** included; download them yourself:

- [SpeechOcean762](https://github.com/k2-fsa/SpeechOcean762) (Apache-2.0): unzip audio to `wavs/speechocean762-1.2.0/WAVE`, put `scores/scores.json` at the repo root as `scores.json`. `train_utt.txt` / `test_utt.txt` (official speaker-disjoint split lists, kaldi `spk2utt` format) are included.
- LibriSpeech `train-clean-100` + `test-clean` (CC-BY-4.0) under `wavs/` — optional, used for cross-domain expansion.

```bash
# 1. features from the frozen encoder (CPU ok, GPU faster)
python extract_features.py

# 2. CTC head on the official split (LibriSpeech expansion via --extra)
python train_gop_head.py --train_ids_file train_utt.txt --test_ids_file test_utt.txt \
    --extra gop_data_libri --out gop_data/ctc_head_official.pt

# 3. XGBoost scorers (prints the results table above)
python train_gop_scorer.py --head gop_data/ctc_head_official.pt \
    --train_ids_file train_utt.txt --test_ids_file test_utt.txt \
    --use_calib --tag official

# 4. robustness check: 5-fold speaker-disjoint CV + grid search
python cv_gop_scorer.py
```

## How the score is computed

1. **logprob-GOP**: for each reference phone, the CTC posterior of the phone id is read off the reference-constrained Viterbi alignment and averaged in log domain over the phone's frame span. (We compared probability-GOP / max-logit / alignment-free variants; logprob won on every level.)
2. **Aggregation**: XGBoost regressors/classifiers take per-phone GOP stats (+ per-phone native-z calibration from LibriSpeech) and produce phone `p_correct`, word score, sentence accuracy and fluency.
3. **Leniency check**: a probe against native speakers confirms the model penalises mispronunciations monotonically (2/1/0 ratings -> mean GOP -0.84/-1.62/-4.34); raw GOP is still shifted upward for non-native speech, hence the calibration features.

## Licenses and third parties

- Code + released weights: Apache-2.0.
- Frozen encoder: Qwen3-ASR (Apache-2.0). You download it separately — it is not part of this repo.
- Annotations/labels redistributed in `gop_data/labels.json` derive from SpeechOcean762 (Apache-2.0).
- LibriSpeech: CC-BY-4.0 (used for training data only; not redistributed).
- **G2P note**: `g2p.py` uses `phonemizer` + `espeak-ng`, which is GPL-3.0. For a strictly permissive stack, swap in a rule-based or dictionary G2P; the rest of the pipeline is Apache-2.0.

## Citation

Coming soon (paper in preparation).
