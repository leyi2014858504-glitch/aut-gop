# AuT-GOP

**Goodness of Pronunciation (GOP) scoring from a frozen lightweight ASR encoder** — no wav2vec2, no fine-tuned LLM, web-deployable.

Pipeline: `audio (16 kHz) -> frozen AuT encoder (Qwen3-ASR-0.6B, ONNX int4) -> single linear CTC head -> reference-constrained Viterbi alignment -> logprob-GOP + posterior-shape features -> gradient-boosted aggregation (XGBoost) -> phone / word / sentence scores`.

Alongside the classic per-phone logprob-GOP, the aggregation consumes 16 **posterior-shape** statistics (frame-level entropy, top1−top2 margin, best competitor-phone mass, pairwise confusability — each aggregated mean/min/max/std over the phone span). These encode *how* the posterior is distributed, not just how sure the reference phone is, and lift every metric (see Phase notes below).

Trainable part is tiny: one linear CTC head (162 KB) + 4 XGBoost scorers (~4 MB). Everything heavier is frozen and shared with the ASR you may already run.

## Results (SpeechOcean762, official 2500/2500 speaker-disjoint split)

Head retrained on the official train split only + LibriSpeech (no test-speaker leakage).

| Level | Baseline (mean logprob-GOP) | AuT-GOP (XGBoost + shape) |
|---|---|---|
| Phone AUC (correct/incorrect) | 0.723 | **0.823** |
| Word Spearman / PCC | 0.395 | 0.394 / **0.628** |
| Sentence accuracy Spearman / PCC | 0.426 | **0.592 / 0.687** |
| Sentence fluency Spearman / PCC | 0.279 | **0.603 / 0.676** |

- CTC head PER on the official test split: 28.0 % (24.7 % on train).
- 5-fold speaker-disjoint CV (in-domain split, robustness): sentence 0.656 ± 0.045, fluency 0.683 ± 0.024, phone AUC 0.829 ± 0.007.
- Sentence-level PCC 0.687 is on par with published fine-tuned systems on SpeechOcean762 (E2E-R ≈ 0.68, GOPT ≈ 0.68; PCC protocol).

## Generalization & ablations

**Cross-encoder** — the identical recipe (frozen encoder → linear CTC head → logprob-GOP + posterior-shape features → XGBoost), same official split, same LibriSpeech expansion + native calibration, on a structurally different encoder, SenseVoice-Small (512-d, non-autoregressive; via `funasr`/PyTorch):

| | Qwen3-AuT (1024-d) | SenseVoice-Small (512-d) |
|---|---|---|
| phone AUC | **0.823** | 0.815 |
| word PCC | **0.628** | 0.624 |
| sentence acc Spearman / PCC | **0.592 / 0.687** | 0.543 / 0.646 |
| XGBoost gain over raw logprob-GOP (sentence) | **+0.166** | +0.035 |

The recipe transfers: phone/word levels match; the aggregation layer's gain is **encoder-dependent** — it heavily rescues the soft, information-rich AuT posteriors (+0.166) and adds little to SenseVoice's already tight-span posteriors (+0.035).

**Head capacity** — same AuT features, same split, same enriched feature set; only the CTC head changes:

| | linear (41K, 162 KB) | DW-CNN (171K, 670 KB) |
|---|---|---|
| CTC head PER | 28.0 % | **21.1 %** |
| phone AUC | 0.823 | 0.826 |
| word PCC | **0.628** | 0.563 |
| sentence acc Spearman / PCC | **0.592 / 0.687** | 0.571 / 0.633 |
| sentence flu PCC | **0.676** | 0.618 |

The sharper head wins at *recognition* (PER 28→21 %) and still **loses at assessment**, even after the posterior-shape features close ~2/3 of its earlier deficit. Post-hoc **temperature scaling cannot explain the gap away**: both heads sit at their NLL optimum (T\* = 1) and GOP correlation degrades monotonically for every T ≠ 1 (`probe_temperature.py`) — the loss is representational, not a global sharpness bias. GOP wants soft posteriors that **retain aggregable uncertainty**, not argmax-sharp ones (`gop_heads.py`, `--head_kind`).

**Posterior-shape features** — with the linear head fixed, adding the 16 shape statistics on top of the logprob-only feature set: phone AUC 0.815 → **0.823**, word PCC 0.611 → **0.628**, sentence Spearman/PCC 0.553/0.682 → **0.592/0.687**, fluency PCC 0.625 → **0.676**; the same direction holds for the CNN head and for SenseVoice (cross-encoder table above). Zero training cost — a feature-pipeline change only.

**Frame rate** — segment-mean logprob-GOP is invariant to frame density by construction; empirically, 4× linear upsampling of features collapses PER (25.6 % → 62.6 %) and fusing the native 25 Hz conv tap (window-aligned, `gop_extract_tap25.py`) is neutral (28.5 % vs 28.6 %). 12.5 Hz is sufficient.

## Quick start

Requirements: Python 3.10+ (developed on 3.14), CPU inference works (`onnxruntime`).

```bash
pip install -r requirements.txt
```

1. Download the frozen encoder from [Qwen3-ASR-0.6B-int4](https://huggingface.co/) and place **only** `config.json` + `encoder.int4.onnx` (~711 MB) under `models/qwen3-asr-0.6b-int4/`. (The decoder is never loaded for GOP.)
2. Score any wav against a reference transcript:

```bash
python score_wav.py --wav my.wav --text "WE CALL IT BEAR" \
    --head ctc_head_official.pt --scorer_tag official_shape
```

Output: sentence accuracy / fluency (0–10), per-word scores, and the most suspicious phones with per-phone `p_correct`.

Library use:

```python
from gop import GOPScorer
from gop_data import load_wav_16k

scorer = GOPScorer(head="ctc_head_official.pt", scorer_tag="official_shape")
result = scorer.score_audio(load_wav_16k("my.wav"), "WE CALL IT BEAR")
# result: {"phones": [...], "words": [...], "sentence": {"accuracy": .., "fluency": ..}}
```

## Repository layout

| File | Role |
|---|---|
| `gop.py` | `GOPScorer` deployable component (encoder ONNX + CTC head + alignment + scorers) |
| `gop_heads.py` | CTC head factory (`linear` / `cnn`), architecture auto-detected on load |
| `gop_data.py` | data helpers: wav loading, splits (kaldi spk2utt or flat), scores |
| `g2p.py` | text -> ARPABET-39 via phonemizer/espeak-ng (IPA bridge) |
| `eval_gop.py` | CTC forced alignment + GOP variants (prob / logprob / max-logit / AF) |
| `extract_sv_features.py` | second-encoder probe: SenseVoice-Small features via funasr (optional deps) |
| `gop_extract_tap25.py` | ONNX graph surgery + 25 Hz conv-tap fusion (frame-rate ablation) |
| `extract_features.py` | dump frozen-encoder features `[T,1024]` for SpeechOcean762 |
| `train_gop_head.py` | train the CTC head (`--head_kind linear|cnn`, CTC loss) |
| `gop_features.py` | build phone/word/sentence GOP feature matrices (+ posterior-shape stats + native-z calibration) |
| `train_gop_scorer.py` | train XGBoost scorers (official or ad-hoc split) |
| `cv_gop_scorer.py` | 5-fold speaker-disjoint CV + hyper-parameter grid |
| `probe_leniency.py` | native vs non-native leniency probe |
| `probe_temperature.py` | temperature-scaling diagnostic (recognition-vs-assessment mechanism) |
| `probe_frames.py` | frame-rate (upsampling) ablation |
| `build_libri_manifest.py`, `extract_libri_features.py`, `eval_libri_per.py` | LibriSpeech cross-domain expansion |
| `gop_data/` | released weights: CTC heads (`ctc_head_official.pt`, `ctc_head_libri.pt`, DW-CNN + SenseVoice heads for the ablations), XGBoost scorers with posterior-shape features (`official_shape`, `cv_best_n200_d4_lr0.05`, `cnn_official_shape`, `sv_shape` tags), `inventory.json`, `labels.json`, `native_stats.npz` |

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
    --use_calib --tag official_shape

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

## Scope & limitations

All scoring results are evaluated on SpeechOcean762, whose speakers are **L1-Mandarin**; generalization of the posterior-calibration / leniency findings to other L1 groups is untested (L2-ARCTIC is the natural next probe). Sentence-level numbers are compared against published systems as reported (protocol differences noted), not re-run.
