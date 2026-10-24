# AI Music Generation Validator

UCU AI course project. Checks whether AI-generated music follows its text prompt and compares
the result with expert judgements.

Data: [SunoCaps](https://www.kaggle.com/datasets/miguelcivit/sunocaps), 256 Suno tracks =
64 MusicCaps prompts × 4 versions (`_1`/`_2` generated from the aspect list, `_3`/`_4` from the
caption), with expert comments about where a track does not match its prompt.

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
| 4. verdicts + evaluation | `run_validation` | in progress |

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
  On the 84 tracks with a tempo word, ROC-AUC for fast vs slow prompts was
  **0.48** (librosa beat tracker), **0.69** (librosa with a flat prior), **0.89** (CLAP).
  CLAP leans towards "slow", so each class probability is divided by its mean over all 256 tracks
  before the argmax (prior correction): satisfied tempo words 40 → 48 of 84.
  See `scripts/compare_tempo.py`; a 24-track listening check (`results/tempo_listening.csv`) is pending.

## Current limitations

- No Essentia models yet (genre / mood / instrument classifiers); those checks fall back to CLAP.
- "Medium" tempo is weak: 15 of 28 medium-prompted tracks still come out as slow after the correction.
- No expert mismatch labels for tempo or mood (0 of 202 records), so those categories can only be
  evaluated by our own listening and the per-track `Emotion` labels.
- The `Instrumental` column is the generator setting, not a listening label, so it is a weak
  ground truth for vocals (CLAP agrees on 82% of tracks, always guessing "vocals" gives 76%).
