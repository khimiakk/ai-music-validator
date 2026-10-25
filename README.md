# AI Music Generation Validator

UCU AI course project. Checks whether AI-generated music follows its text prompt and compares
the result with expert judgements.

Data: [SunoCaps](https://www.kaggle.com/datasets/miguelcivit/sunocaps), 256 Suno tracks =
64 MusicCaps prompts × 4 versions (`_1`/`_2` generated from the aspect list, `_3`/`_4` from the
caption), with expert comments about where a track does not match its prompt.

**Midterm:** results notebook [`notebooks/03_first_results.ipynb`](notebooks/03_first_results.ipynb) ·
status, what didn't work and the plan until the final [`docs/midterm_status.md`](docs/midterm_status.md) ·
all metric tables [`results/summary.md`](results/summary.md)

## What it does

1. Parse each prompt into checkable requirements in 6 categories:
   **genre, instruments, vocals, tempo, mood, other** (e.g. "fast tempo", "female vocals", "no strings").
2. Check every requirement against the audio and give a verdict:

   | verdict | meaning |
   |---|---|
   | `satisfied` | the requirement is met |
   | `ignored` | a requested element is missing (or a prohibited one is present) |
   | `changed` | the element is there but different (e.g. slow instead of fast) |
   | `hallucinated` | something not in the prompt was added |

3. Compare the verdicts with the expert comments (turned into per-category ground truth).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d miguelcivit/sunocaps -p data/raw --unzip
```

Expected layout (`data/` is not in git):

```
data/raw/SunoCaps_dataset.csv
data/raw/audio/<track_id>.mp3     # 256 files; subfolders are fine, aimv.data.audio_path searches them
```

## How to run (in order)

| step | command | status |
|---|---|---|
| 1. audio features (tempo + CLAP embeddings, ~70 s on Apple-silicon MPS) | `python scripts/extract_audio_features.py` | done |
| 2. expert ground truth | `python scripts/build_ground_truth.py` (`--review` lists drafts) | done; 62 of 213 records still need review |
| 3. prompt requirements | `python scripts/parse_prompts.py` (`--examples` prints 8 prompts side by side) | done; 2,172 requirements, covers 96% of expert records by category, 92% by element |
| 4. verdicts + evaluation | `python scripts/run_validation.py` (~30 s, cached embeddings only) | done; see `results/summary.md` |
| 5. Essentia scores (optional, Google Colab) | `notebooks/04_essentia.ipynb` → copy `essentia_scores.csv` to `data/processed/`, rerun step 4 | done (~6.5 s per track in Colab); compared with CLAP |

Step 5 is optional: without `data/processed/essentia_scores.csv`, step 4 uses CLAP for every check and
skips the CLAP vs Essentia comparison (section 10 of `results/summary.md`).

Sanity checks after step 1: `python scripts/check_audio_features.py` and
`python scripts/compare_tempo.py` (writes to `results/`). Tests: `pytest`.

## Ground truth (expert comments)

The 183 expert comments become 213 mismatch records (`src/aimv/annotations.py`): the 19 notebook-02
rules, manual review batches 01/02, draft rules 03 and manual review 04 (all text-reviewed).
62 records still need review (54 draft, 8 need an audio check).

Change vs notebook 02: the rule_v2 records "not instrumental" / "does not follow instrumental" mean
the prompt asked for no vocals and the track has vocals, the same as "added voice", so their
error_type is `added` (verdict `ignored`) instead of `incorrect` (`changed`). The notebook-02 rules
alone still give 117 records on 104 tracks; 18 of them now carry source `manual_review_04`.

## Model choices

- **CLAP: `laion/larger_clap_general`** (3 × 10 s windows per track, averaged), for zero-shot checks.
  The HF checkpoint `laion/larger_clap_music` produced near-identical embeddings in our setup
  (tested with two transformers versions), so we use the general LAION CLAP; the original music
  checkpoint via the `laion_clap` package is a later check.
- **Explicit BPM** ("120 bpm"): librosa beat tracker, ±8%, half/double tempo accepted.
- **Tempo words** ("slow", "fast", ...): CLAP zero-shot over *slow / medium / fast tempo music*.
  On the 88 tracks with a tempo word, ROC-AUC for fast vs slow prompts was
  **0.47** (librosa beat tracker), **0.69** (librosa with a flat prior), **0.90** (CLAP).
  CLAP leans towards "slow", so each class probability is divided by its mean over all 256 tracks
  before the argmax (prior correction): satisfied tempo words 42 → 50 of 88.
  See `scripts/compare_tempo.py`; a 24-track listening check (`results/tempo_listening.csv`) is pending.
- **Verdicts** (`src/aimv/verdicts.py`): vocal presence and gender by CLAP zero-shot (p ≥ 0.5);
  every other requirement by CLAP similarity, calibrated as the percentile of the track's score among
  tracks whose prompt does not mention that element (satisfied if ≥ tau = 0.5; prohibitions reversed);
  vocals heard in a track whose prompt says nothing about vocals → `hallucinated`.
- **Evaluation** (`src/aimv/evaluation.py`): 95% CIs from a cluster bootstrap over `original_id`
  (1000 resamples); thresholds and tau are chosen with 5-fold GroupKFold over `original_id`.

## Results (details in `results/summary.md`)

Main result = commented tracks (subset A), per (track, category) cell, with 95% CIs:

| category | expert positives | F1 | ROC-AUC | trivial baseline F1 |
|---|---|---|---|---|
| vocals | 70 | 0.75 [0.62, 0.85] | 0.84 [0.75, 0.91] | 0.55 |
| other | 74 | 0.64 [0.50, 0.75] | 0.76 [0.66, 0.85] | 0.70 |
| instruments | 38 | 0.36 [0.20, 0.51] | 0.55 [0.38, 0.71] | 0.35 |
| genre | 20 | 0.32 [0.06, 0.53] | 0.74 [0.59, 0.88] | 0.27 |
| micro (these 4) | 202 | 0.58 [0.50, 0.65] | 0.76 [0.70, 0.82] | 0.48 |

- Tempo and mood: no expert labels, so they are not evaluated against the experts.
- Track level (all 256 tracks, any mismatch): ROC-AUC 0.64 [0.54, 0.74] for the modular system vs
  0.58 for global CLAP prompt similarity and 0.49 for the trivial baseline.
- tau chosen by GroupKFold: 0.55 in every fold (micro F1 0.59 vs 0.58 at tau = 0.5).
- Vocal presence vs the labels: accuracy 0.82; emotion zero-shot (8 classes): accuracy 0.32,
  macro-F1 0.23 (majority class: 0.36 / 0.07).
- Aspect list vs caption: no difference in the share of satisfied requirements (0.70 vs 0.69, paired).

**CLAP vs Essentia** (same rules and calibration; paired difference Essentia − CLAP with 95% CIs):

| | CLAP | Essentia | Essentia − CLAP |
|---|---|---|---|
| genre ROC-AUC | 0.74 | 0.73 | −0.01 [−0.09, 0.07] |
| instruments ROC-AUC | 0.55 | 0.56 | +0.01 [−0.11, 0.10] |
| vocals ROC-AUC | 0.84 | 0.88 | +0.04 [−0.04, 0.13] |
| micro ROC-AUC | 0.76 | 0.77 | +0.01 [−0.02, 0.04] |
| track-level ROC-AUC (all 256) | 0.64 | 0.62 | −0.02 [−0.09, 0.05] |
| vocal presence accuracy | 0.82 | **0.92** | **+0.10 [0.01, 0.18]** |

The Essentia voice model is clearly better for vocal presence; everywhere else the two models (and their
average) are not significantly different. The main pipeline still uses CLAP; full tables in
`results/summary.md` §10 and notebook 03 §3.7.

## Current limitations

- Essentia does not improve genre, instruments or mood over CLAP; only its voice model helps (vocal presence).
- Instruments are the weakest check with both models (ROC-AUC 0.55 CLAP, 0.56 Essentia), so the problem is
  probably the setup rather than the audio model: many songs contain bass, drums or keyboards even when the
  prompt does not mention them (noisy calibration), and a missing instrument is hard to confirm.
- Only 13 of the 183 commented tracks have no expert mismatch, so track-level results on subset A
  are very uncertain; all 256 tracks (no comment = OK) give the more useful track-level comparison.
- "Medium" tempo is weak: 17 of 30 medium-prompted tracks still come out as slow after the correction.
- No expert mismatch labels for tempo or mood (0 of 213 records), so those categories can only be
  evaluated by our own listening and the per-track `Emotion` labels.
- The `Instrumental` column is the generator setting, not a listening label, so it is a weak
  ground truth for vocals (CLAP agrees on 82% of tracks, always guessing "vocals" gives 76%).
  The 46 disagreements are in `results/vocals_listening.csv`; like `results/tempo_listening.csv`,
  it still has to be labelled by listening.
