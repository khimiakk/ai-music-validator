"""Compare tempo estimators on tracks whose prompt has a tempo word (fast / slow / medium ...).

    compare_tempo.py                     # comparison + listening sheet
    compare_tempo.py --score-listening   # score the estimators against my_label in tempo_listening.csv

    a) bpm        librosa beat tracker (audio_features.csv, default prior centred on 120 BPM)
    b) bpm_flat   librosa tempo with a flat prior over 40-220 BPM (extra pass, added to audio_features.csv)
    c) clap       CLAP zero-shot "slow / medium / fast tempo music", score = p_fast - p_slow, verdict = argmax
    d) clap_corr  same, but each class probability is divided by its mean over all 256 tracks first
                  (CLAP leans towards "slow"); nothing here is tuned on the listening labels

    Verdicts: a) and b) use tempo_verdict with the word's BPM range, c) and d) use tempo_word_verdict.

    results/tempo_comparison.csv          one row per track
    results/tempo_comparison_summary.csv  per estimator: medians, ROC-AUC fast vs slow, satisfied counts per word
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
                              estimate_tempo_flat_prior, prior_correct, tempo_verdict,
                              tempo_word_verdict)
from aimv.data import PROCESSED_DIR, PROJECT_ROOT, audio_path, load_sunocaps  # noqa: E402
from aimv.prompt_parsing import parse_dataframe  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_CSV = PROCESSED_DIR / "audio_features.csv"
COMPARISON_CSV = RESULTS_DIR / "tempo_comparison.csv"
LISTENING_CSV = RESULTS_DIR / "tempo_listening.csv"

ESTIMATORS = {"bpm": "a) librosa beat tracker",
              "bpm_flat": "b) librosa, flat prior",
              "clap": "c) CLAP argmax",
              "clap_corr": "d) CLAP prior-corrected"}
SCORE_LABELS = {"bpm": "BPM", "bpm_flat": "BPM", "clap": "p(fast) - p(slow)", "clap_corr": "p(fast) - p(slow)"}
P_COLS = [f"p_{c}" for c in TEMPO_CLASSES]
GROUPS = ["slow", "slow/medium", "medium", "medium/fast", "fast"]
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
    t = reqs[(reqs["category"] == "tempo") & reqs["value"].isin(TEMPO_WORD_CLASSES)].drop_duplicates("track_id")
    t = t.assign(bpm_min=t["params"].str["bpm_min"], bpm_max=t["params"].str["bpm_max"])
    t = t[["track_id", "original_id", "prompt_type", "value", "bpm_min", "bpm_max"]].rename(columns={"value": "tempo_word"})
    t["group"] = t["tempo_word"].map(lambda w: "/".join(TEMPO_WORD_CLASSES[w]))
    return t.reset_index(drop=True)


def clap_tempo_probs() -> pd.DataFrame:
    """CLAP probabilities for slow / medium / fast tempo music, for every track."""
    meta = json.load(open(PROCESSED_DIR / "clap_meta.json"))
    embs = np.load(PROCESSED_DIR / "clap_audio_windows.npz")
    scorer = ClapScorer(meta["model"])
    text_embs = scorer.embed_text([f"{c} tempo music" for c in TEMPO_CLASSES])
    probs = [zero_shot_probs(track_embedding(embs[tid]), text_embs, meta["logit_scale"]) for tid in embs.files]
    return pd.DataFrame(probs, columns=P_COLS).assign(track_id=embs.files)


def add_verdicts(t: pd.DataFrame, class_means: np.ndarray) -> pd.DataFrame:
    for col in ("bpm", "bpm_flat"):
        t[f"verdict_{col}"] = [tempo_verdict(b, lo, hi)[0] for b, lo, hi in zip(t[col], t["bpm_min"], t["bpm_max"])]
    probs = t[P_COLS].to_numpy()
    t["verdict_clap"] = [tempo_word_verdict(p, w)[0] for p, w in zip(probs, t["tempo_word"])]
    t["verdict_clap_corr"] = [tempo_word_verdict(p, w, class_means)[0] for p, w in zip(probs, t["tempo_word"])]
    return t


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    fast_slow = t[t["group"].isin(["fast", "slow"])]
    is_fast = fast_slow["group"] == "fast"
    words = t.sort_values("group", key=lambda g: g.map(GROUPS.index))["tempo_word"].unique()
    rows = []
    for key, name in ESTIMATORS.items():
        row = dict(estimator=name, median_fast=fast_slow.loc[is_fast, key].median(),
                   median_slow=fast_slow.loc[~is_fast, key].median(),
                   auc_fast_vs_slow=roc_auc_score(is_fast, fast_slow[key]))
        for w in words:
            sub = t[t["tempo_word"] == w]
            row[f"satisfied {w} (n={len(sub)})"] = int((sub[f"verdict_{key}"] == "satisfied").sum())
        row[f"satisfied total (n={len(t)})"] = int((t[f"verdict_{key}"] == "satisfied").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame, summary: pd.DataFrame, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in GROUPS if (t["group"] == g).any()]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="#fcfcfb")
    rng = np.random.default_rng(0)
    for ax, (key, name), auc in zip(axes.flat, ESTIMATORS.items(), summary["auc_fast_vs_slow"]):
        data = [t.loc[t["group"] == g, key] for g in groups]
        ax.set_facecolor("#fcfcfb")
        ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#dbe8f8", edgecolor="#2a78d6"), medianprops=dict(color="#0b0b0b"),
                   whiskerprops=dict(color="#2a78d6"), capprops=dict(color="#2a78d6"))
        for i, d in enumerate(data, 1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=12, color="#2a78d6", alpha=0.6, zorder=3)
        ax.set_xticks(range(1, len(groups) + 1), [f"{g}\n(n={len(d)})" for g, d in zip(groups, data)])
        ax.set_title(f"{name}: {SCORE_LABELS[key]}\nAUC fast vs slow = {auc:.2f}", loc="left", fontsize=10,
                     color="#0b0b0b")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", length=0)
    fig.supxlabel("tracks grouped by the prompt's tempo word", color="#52514e", fontsize=10)
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


def top_class(probs: np.ndarray) -> list[str]:
    return [TEMPO_CLASSES[i] for i in probs.argmax(axis=1)]


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
        ESTIMATORS["clap"]: top_class(t[P_COLS].to_numpy()),
        ESTIMATORS["clap_corr"]: top_class(t[[c + "_corr" for c in P_COLS]].to_numpy()),
    }
    labels = t["my_label"].to_numpy()
    fast_slow = np.isin(labels, ["fast", "slow"])
    print(f"{len(t)} labelled tracks:", t["my_label"].value_counts().to_dict())
    for name, pred in predictions.items():
        correct = np.asarray(pred) == labels
        line = f"  {name:<26} accuracy {correct.mean():6.1%}"
        if fast_slow.any():
            line += f"   on tracks labelled fast/slow (n={fast_slow.sum()}) {correct[fast_slow].mean():6.1%}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-listening", action="store_true")
    args = ap.parse_args()
    if args.score_listening:
        score_listening()
        return

    df = load_sunocaps()
    feats = add_bpm_flat(pd.read_csv(FEATURES_CSV))
    probs_all = clap_tempo_probs()
    class_means = probs_all[P_COLS].mean().to_numpy()
    print(f"mean CLAP probability over all {len(probs_all)} tracks:",
          {c: round(float(m), 3) for c, m in zip(TEMPO_CLASSES, class_means)})

    t = tempo_word_tracks(df).merge(feats[["track_id", "bpm", "bpm_flat"]], on="track_id")
    t = t.merge(probs_all, on="track_id")
    corrected = np.stack([prior_correct(p, class_means) for p in t[P_COLS].to_numpy()])
    t[[c + "_corr" for c in P_COLS]] = corrected
    t["clap"] = t["p_fast"] - t["p_slow"]
    t["clap_corr"] = t["p_fast_corr"] - t["p_slow_corr"]
    t = add_verdicts(t, class_means)
    print(f"{len(t)} tracks with a tempo word:", t["group"].value_counts().to_dict())

    RESULTS_DIR.mkdir(exist_ok=True)
    t.to_csv(COMPARISON_CSV, index=False)
    summary = summarize(t)
    summary.to_csv(RESULTS_DIR / "tempo_comparison_summary.csv", index=False)
    print(summary.set_index("estimator").T.to_string(float_format=lambda x: f"{x:.3g}"))
    plot(t, summary, RESULTS_DIR / "figures" / "tempo_comparison.png")
    make_listening_sheet(t)


if __name__ == "__main__":
    main()
