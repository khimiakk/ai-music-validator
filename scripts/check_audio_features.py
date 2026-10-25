"""sanity checks on the output of extract_audio_features.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimv.audio.clap import ClapScorer, track_embedding, zero_shot_probs  # noqa: E402
from aimv.data import PROCESSED_DIR, PROJECT_ROOT, load_sunocaps  # noqa: E402

FIG_DIR = PROJECT_ROOT / "results" / "figures"
VOCAL_TEXTS = ["a song with singing vocals", "instrumental music without vocals"]


def check_coverage(df, feats, embs):
    print("== 1. coverage ==")
    ids = set(df["track_id"])
    for name, have in (("audio_features.csv", set(feats["track_id"])), ("clap_audio_windows.npz", set(embs))):
        missing = sorted(ids - have)
        print(f"{name}: {len(ids & have)}/{len(ids)} tracks" + (f", missing e.g. {missing[:5]}" if missing else ""))


def check_tempo(df, feats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n== 2. tempo ==")
    t = df.merge(feats, on="track_id")
    print(t["bpm"].describe().round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.hist(t["bpm"], bins=range(40, 230, 10), color="#2a78d6", edgecolor="#fcfcfb", linewidth=2)
    ax.set_title(f"Estimated tempo of the {len(t)} SunoCaps tracks", loc="left", color="#0b0b0b")
    ax.set_xlabel("BPM (librosa beat tracker)", color="#52514e")
    ax.set_ylabel("tracks", color="#52514e")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", length=0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "tempo_hist.png", dpi=150, bbox_inches="tight")
    print(f"saved {FIG_DIR / 'tempo_hist.png'}")

    prompt = t["actual_prompt"].str.lower()
    for word in ("fast", "slow"):
        group = t[prompt.str.contains(rf"\b{word}\b")]
        print(f'prompt contains "{word}": {len(group)} tracks, median BPM {group["bpm"].median():.1f}')


def check_vocals(df, embs, scorer, logit_scale):
    print("\n== 3. CLAP vocals sanity ==")
    text_embs = scorer.embed_text(VOCAL_TEXTS)
    p_vocals = [zero_shot_probs(track_embedding(embs[tid]), text_embs, logit_scale)[0] for tid in df["track_id"]]
    v = pd.DataFrame({"prompt_type": df["prompt_type"], "p_vocals": p_vocals,
                      "vocals_expected": df["Instrumental"].astype(str).str.lower().eq("false")})
    v["vocals_predicted"] = v["p_vocals"] > 0.5
    agree = (v["vocals_predicted"] == v["vocals_expected"]).mean()
    baseline = max(v["vocals_expected"].mean(), 1 - v["vocals_expected"].mean())
    print(f"agreement with Instrumental column: {agree:.1%} (always predicting the majority class: {baseline:.1%})")
    print(pd.crosstab(v["vocals_expected"], v["vocals_predicted"],
                      rownames=["vocals expected"], colnames=["CLAP predicts vocals"]).to_string())
    print("mean p(vocals) by expected label:", v.groupby("vocals_expected")["p_vocals"].mean().round(3).to_dict())


def check_retrieval(df, embs, scorer):
    print("\n== 4. CLAP retrieval sanity (own prompt among the 64 of the same type) ==")
    for prompt_type, part in df.groupby("prompt_type"):
        candidates = part.drop_duplicates("original_id")
        text_embs = scorer.embed_text(candidates["actual_prompt"].tolist())
        position = {oid: i for i, oid in enumerate(candidates["original_id"])}
        ranks = []
        for _, row in part.iterrows():
            sims = text_embs @ track_embedding(embs[row["track_id"]])
            own = sims[position[row["original_id"]]]
            ranks.append(int((sims > own).sum()) + 1)
        ranks = np.array(ranks)
        print(f"{prompt_type:<12} {len(ranks)} tracks, {len(candidates)} prompts: R@1 {np.mean(ranks == 1):.1%}, "
              f"R@5 {np.mean(ranks <= 5):.1%}, median rank {np.median(ranks):.1f} (random ~ {(len(candidates) + 1) / 2:.1f})")


def main():
    df = load_sunocaps()
    feats = pd.read_csv(PROCESSED_DIR / "audio_features.csv")
    embs = dict(np.load(PROCESSED_DIR / "clap_audio_windows.npz"))
    meta = json.load(open(PROCESSED_DIR / "clap_meta.json"))

    check_coverage(df, feats, embs)
    check_tempo(df, feats)

    df = df[df["track_id"].isin(embs)]
    scorer = ClapScorer(meta["model"])  # text embeddings must use the audio embeddings' model
    print(f"\nCLAP {meta['model']}, logit_scale {meta['logit_scale']:.2f} (from clap_meta.json)")
    check_vocals(df, embs, scorer, meta["logit_scale"])
    check_retrieval(df, embs, scorer)


if __name__ == "__main__":
    main()
