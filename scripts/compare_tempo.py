"""Compare three tempo estimators on tracks whose prompt has a tempo word (fast / slow / medium ...).

    compare_tempo.py                     # comparison + listening sheet
    compare_tempo.py --score-listening   # score the estimators against my_label in tempo_listening.csv

    a) bpm       librosa beat tracker (audio_features.csv, default prior centred on 120 BPM)
    b) bpm_flat  librosa tempo with a flat prior over 40-220 BPM (extra pass, added to audio_features.csv)
    c) clap      CLAP zero-shot "slow / medium / fast tempo music", score = p_fast - p_slow

    results/tempo_comparison.csv          one row per track
    results/tempo_comparison_summary.csv  medians + ROC-AUC fast vs slow per estimator
    results/figures/tempo_comparison.png  boxplots
    results/tempo_listening.csv           24 tracks to label by ear (created once, never overwritten)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimv.audio.clap import ClapScorer, track_embedding, zero_shot_probs  # noqa: E402
from aimv.audio.io import load_audio  # noqa: E402
from aimv.audio.tempo import (TEMPO_CLASSES, TEMPO_SR, TEMPO_WORD_CLASSES,  # noqa: E402
                              estimate_tempo_flat_prior)
from aimv.data import PROCESSED_DIR, PROJECT_ROOT, audio_path, load_sunocaps  # noqa: E402
from aimv.prompt_parsing import parse_dataframe  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_CSV = PROCESSED_DIR / "audio_features.csv"
COMPARISON_CSV = RESULTS_DIR / "tempo_comparison.csv"
LISTENING_CSV = RESULTS_DIR / "tempo_listening.csv"

ESTIMATORS = {"bpm": "a) librosa beat tracker (BPM)",
              "bpm_flat": "b) librosa, flat prior (BPM)",
              "clap_score": "c) CLAP  p(fast) - p(slow)"}
GROUPS = ["slow", "medium", "medium/fast", "fast"]
SLOW_BELOW, FAST_FROM = 90, 120  # BPM -> class, from the slow / medium / fast ranges in vocab.TEMPO_WORDS


def add_bpm_flat(feats: pd.DataFrame) -> pd.DataFrame:
    """Extra pass over the audio: flat-prior tempo for every track, saved into audio_features.csv."""
    if "bpm_flat" in feats and feats["bpm_flat"].notna().all():
        return feats
    print(f"computing bpm_flat for {len(feats)} tracks ...", flush=True)
    feats["bpm_flat"] = [estimate_tempo_flat_prior(load_audio(audio_path(tid), sr=TEMPO_SR), TEMPO_SR)
                         for tid in feats["track_id"]]
    feats.to_csv(FEATURES_CSV, index=False)
    return feats


def tempo_word_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per track whose parsed requirements contain a tempo word (explicit BPM values are skipped)."""
    reqs = parse_dataframe(df)
    t = reqs[(reqs["category"] == "tempo") & reqs["value"].isin(TEMPO_WORD_CLASSES)]
    t = t.drop_duplicates("track_id")[["track_id", "original_id", "prompt_type", "value"]]
    t = t.rename(columns={"value": "tempo_word"})
    t["group"] = t["tempo_word"].map(lambda w: "/".join(TEMPO_WORD_CLASSES[w]))
    return t.reset_index(drop=True)


def add_clap_probs(t: pd.DataFrame) -> pd.DataFrame:
    meta = json.load(open(PROCESSED_DIR / "clap_meta.json"))
    embs = np.load(PROCESSED_DIR / "clap_audio_windows.npz")
    scorer = ClapScorer(meta["model"])
    text_embs = scorer.embed_text([f"{c} tempo music" for c in TEMPO_CLASSES])
    probs = np.stack([zero_shot_probs(track_embedding(embs[tid]), text_embs, meta["logit_scale"])
                      for tid in t["track_id"]])
    for i, c in enumerate(TEMPO_CLASSES):
        t[f"p_{c}"] = probs[:, i]
    t["clap_score"] = t["p_fast"] - t["p_slow"]
    return t


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    fast_slow = t[t["group"].isin(["fast", "slow"])]
    is_fast = fast_slow["group"] == "fast"
    rows = []
    for col, name in ESTIMATORS.items():
        rows.append(dict(estimator=name, median_fast=fast_slow.loc[is_fast, col].median(),
                         median_slow=fast_slow.loc[~is_fast, col].median(),
                         auc_fast_vs_slow=roc_auc_score(is_fast, fast_slow[col])))
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame, summary: pd.DataFrame, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in GROUPS if (t["group"] == g).any()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")
    rng = np.random.default_rng(0)
    for ax, (col, name), auc in zip(axes, ESTIMATORS.items(), summary["auc_fast_vs_slow"]):
        data = [t.loc[t["group"] == g, col] for g in groups]
        ax.set_facecolor("#fcfcfb")
        ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#dbe8f8", edgecolor="#2a78d6"), medianprops=dict(color="#0b0b0b"),
                   whiskerprops=dict(color="#2a78d6"), capprops=dict(color="#2a78d6"))
        for i, d in enumerate(data, 1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=12, color="#2a78d6", alpha=0.6, zorder=3)
        ax.set_xticks(range(1, len(groups) + 1), [f"{g}\n(n={len(d)})" for g, d in zip(groups, data)])
        ax.set_title(f"{name}\nAUC fast vs slow = {auc:.2f}", loc="left", fontsize=10, color="#0b0b0b")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", length=0)
    axes[0].set_ylabel("value for tracks grouped by the prompt's tempo word", color="#52514e")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def make_listening_sheet(t: pd.DataFrame, n_per_group: int = 12, seed: int = 0):
    """12 fast- and 12 slow-prompted tracks, one version per original_id first.

    "slow" comes from only 7 prompts and "fast" from 10, so the groups are topped up
    with a second (different) version of an already chosen prompt.
    """
    if LISTENING_CSV.exists():
        print(f"{LISTENING_CSV} already exists, not overwriting it")
        return
    picks = []
    for group in ("fast", "slow"):
        g = t[t["group"] == group].sample(frac=1, random_state=seed)
        first = g.drop_duplicates("original_id")
        second = g.drop(first.index).drop_duplicates("original_id")
        picks.append(pd.concat([first, second]).head(n_per_group))
    sheet = pd.concat(picks).sample(frac=1, random_state=seed)  # mix fast and slow
    sheet = sheet[["track_id", "tempo_word"]].rename(columns={"tempo_word": "prompt_tempo_word"})
    sheet["my_label"] = ""
    sheet.to_csv(LISTENING_CSV, index=False)
    print(f"saved {LISTENING_CSV} ({len(sheet)} tracks, "
          f"{sheet['track_id'].str.rsplit('_', n=1).str[0].nunique()} distinct prompts)")


def bpm_class(bpm: float) -> str:
    return "slow" if bpm < SLOW_BELOW else "fast" if bpm >= FAST_FROM else "medium"


def score_listening():
    """Accuracy of each estimator (and of the prompt word itself) against my_label."""
    sheet = pd.read_csv(LISTENING_CSV, dtype=str).fillna("")
    sheet["my_label"] = sheet["my_label"].str.strip().str.lower()
    sheet = sheet[sheet["my_label"].isin(TEMPO_CLASSES)]
    if sheet.empty:
        print(f"my_label in {LISTENING_CSV} is not filled yet (use slow / medium / fast)")
        return
    t = sheet.merge(pd.read_csv(COMPARISON_CSV), on="track_id")
    predictions = {
        "prompt word": t["group"],
        ESTIMATORS["bpm"]: t["bpm"].map(bpm_class),
        ESTIMATORS["bpm_flat"]: t["bpm_flat"].map(bpm_class),
        "c) CLAP top class": t[[f"p_{c}" for c in TEMPO_CLASSES]].to_numpy().argmax(axis=1),
    }
    predictions["c) CLAP top class"] = [TEMPO_CLASSES[i] for i in predictions["c) CLAP top class"]]
    print(f"{len(t)} labelled tracks:", t["my_label"].value_counts().to_dict())
    for name, pred in predictions.items():
        print(f"  {name:<32} accuracy {np.mean(np.asarray(pred) == t['my_label'].to_numpy()):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-listening", action="store_true")
    args = ap.parse_args()
    if args.score_listening:
        score_listening()
        return

    df = load_sunocaps()
    feats = add_bpm_flat(pd.read_csv(FEATURES_CSV))
    t = tempo_word_tracks(df).merge(feats[["track_id", "bpm", "bpm_flat"]], on="track_id")
    t = add_clap_probs(t)
    print(f"{len(t)} tracks with a tempo word:", t["group"].value_counts().to_dict())

    RESULTS_DIR.mkdir(exist_ok=True)
    t.to_csv(COMPARISON_CSV, index=False)
    summary = summarize(t)
    summary.to_csv(RESULTS_DIR / "tempo_comparison_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    plot(t, summary, RESULTS_DIR / "figures" / "tempo_comparison.png")
    make_listening_sheet(t)


if __name__ == "__main__":
    main()
