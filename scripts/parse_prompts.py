"""Stage 3: prompt -> checkable requirements (no audio needed).

    parse_prompts.py              # parse, save, summary + coverage vs the expert records
    parse_prompts.py --examples   # also print 8 random prompts next to their requirements

    data/processed/requirements.csv   one row per (track, requirement); params as a JSON string

Coverage vs experts uses data/processed/mismatch_records.csv (run build_ground_truth.py first):
    a) category level: the track has any requirement of the expert record's category
    b) element level:  a requirement of that category has a key word of the expert's "expected"
                       in its value or phrase (simple word matching). Prohibitions ("no ...",
                       "instrumental only") need a requirement with polarity -1 instead, and
                       generic expectations ("prompted style") only the category.
    An element-level miss is a parser miss if a key word appears in the prompt text, otherwise
    the prompt itself does not name the element.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimv import CATEGORIES, vocab  # noqa: E402
from aimv.audio.tempo import TEMPO_WORD_CLASSES  # noqa: E402
from aimv.data import PROCESSED_DIR, load_sunocaps  # noqa: E402
from aimv.prompt_parsing import parse_dataframe  # noqa: E402

# expert expectations that no prompt-word parser can produce (not counted as parser misses)
NOT_FROM_PROMPT_WORDS = {
    "composition as described (what happens in which bar)": "composition / structure",
    "violins in eighth-note pattern": "rhythmic figure",
}
STOP_WORDS = {"a", "an", "and", "or", "the", "of", "in", "as", "with", "no", "not", "only", "type", "style"}
GENERIC_WORDS = {"vocal", "music", "recording", "sound", "instrument"}  # only used if nothing more specific is left


def words(text) -> set[str]:
    """Lower-case words with a trailing plural "s" removed ("strings" -> "string")."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z]+", str(text).lower())}


def key_words(expected: str) -> set[str]:
    keys = words(expected) - STOP_WORDS
    return (keys - GENERIC_WORDS) or keys


def unique_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """One track per distinct prompt (_1/_2 and _3/_4 share a prompt): 64 aspect lists + 64 captions."""
    return df.drop_duplicates(["original_id", "prompt_type"])


def print_summary(df: pd.DataFrame, reqs: pd.DataFrame):
    first = unique_prompts(df)
    u = reqs[reqs["track_id"].isin(first["track_id"])]
    print(f"(statistics over the {len(first)} distinct prompts)\n")

    print("== requirements per prompt ==")
    n = first.set_index("track_id")["prompt_type"].to_frame().join(u.groupby("track_id").size().rename("n"))
    print(n.fillna(0).groupby("prompt_type")["n"].agg(["mean", "min", "max"]).round(1).to_string())

    print("\n== category x check ==")
    print(pd.crosstab(u["category"], u["check"], margins=True).to_string())

    for cat in CATEGORIES:
        top = u[u["category"] == cat]["value"].value_counts().head(20)
        print(f"\n== top values: {cat} ==\n  " + ", ".join(f"{v} ({c})" for v, c in top.items()))

    tempo = u[u["category"] == "tempo"]
    explicit = tempo["value"].str.endswith("bpm")
    print(f"\n== tempo: {(~explicit).sum()} word requirements, {explicit.sum()} explicit BPM ==")
    print("  words:", tempo.loc[~explicit, "value"].value_counts().to_dict())
    unmapped = sorted(set(tempo.loc[~explicit, "value"]) - set(TEMPO_WORD_CLASSES))
    print("  words tempo_word_verdict does not map:", unmapped or "none")

    fallback = u[(u["category"] == "other") & (u["value"] == u["phrase"]) & ~u["value"].isin(vocab.OTHER_TERMS)]
    print(f"\n== aspect tags in 'other' only because nothing in vocab.py matched: {fallback['value'].nunique()} ==")
    print("  " + ", ".join(sorted(fallback["value"].unique())))


def print_examples(df: pd.DataFrame, reqs: pd.DataFrame, n: int = 8, seed: int = 0):
    ids = pd.Series(df["original_id"].unique()).sample(n, random_state=seed)
    for oid in ids:
        rows = df[df["original_id"] == oid].drop_duplicates("prompt_type").set_index("prompt_type")
        print(f"\n{'=' * 110}\n{oid}")
        print(f"ASPECT LIST: {rows.loc['aspect_list', 'actual_prompt']}")
        print(f"CAPTION:     {rows.loc['caption', 'actual_prompt']}")
        sides = []
        for pt in ("aspect_list", "caption"):
            r = reqs[reqs["track_id"] == rows.loc[pt, "track_id"]]
            sides.append([f"{'-' if p < 0 else ' '}{c}: {v}  [{k}]" for c, v, p, k in
                          zip(r["category"], r["value"], r["polarity"], r["check"])])
        print(f"{'-- from aspect list':<55}-- from caption")
        for i in range(max(map(len, sides))):
            left, right = (s[i] if i < len(s) else "" for s in sides)
            print(f"{left:<55}{right}")


def element_hit(expected: str, same_cat: pd.DataFrame) -> bool:
    """Does a requirement of the expert record's category match the expert's "expected"?"""
    if expected.startswith("prompted"):  # "prompted style": the expert names no element
        return len(same_cat) > 0
    if expected.startswith("no ") or "instrumental" in expected:  # prohibition
        return bool((same_cat["polarity"] < 0).any())
    keys = key_words(expected)
    return any(keys & (words(v) | words(p)) for v, p in zip(same_cat["value"], same_cat["phrase"]))


def print_coverage(df: pd.DataFrame, reqs: pd.DataFrame, records: pd.DataFrame):
    records = records[records["error_type"] != "unexpected_addition"]
    skip = records["expected"].isin(NOT_FROM_PROMPT_WORDS)
    by_track = {tid: g for tid, g in reqs.groupby("track_id")}
    prompt = df.set_index("track_id")["actual_prompt"]

    rows = []
    for _, rec in records[~skip].iterrows():
        r = by_track.get(rec["track_id"], reqs.iloc[:0])
        same_cat = r[r["category"] == rec["category"]]
        rows.append(dict(track_id=rec["track_id"], category=rec["category"], expected=rec["expected"],
                         category_hit=len(same_cat) > 0, element_hit=element_hit(rec["expected"], same_cat),
                         in_prompt=bool(key_words(rec["expected"]) & words(prompt[rec["track_id"]])),
                         parsed=", ".join(sorted(set(same_cat["value"]))) or "-"))
    cov = pd.DataFrame(rows)

    print(f"\n== coverage vs {len(records)} expert records (unexpected_addition skipped) ==")
    summary = cov.groupby("category").agg(records=("track_id", "size"), category_level=("category_hit", "mean"),
                                          element_level=("element_hit", "mean"))
    summary.loc["all"] = [len(cov), cov["category_hit"].mean(), cov["element_hit"].mean()]
    summary[["category_level", "element_level"]] = (summary[["category_level", "element_level"]] * 100).round(1)
    print(summary.to_string())

    misses = cov[~cov["element_hit"]]
    for title, part in (("parser misses (a key word is in the prompt)", misses[misses["in_prompt"]]),
                        ("the prompt does not name the element", misses[~misses["in_prompt"]])):
        print(f"\n== element-level misses, {title}: {len(part)} records ==")
        for (cat, expected), g in part.groupby(["category", "expected"]):
            print(f"  {cat}: {expected!r} ({', '.join(g['track_id'])}) "
                  f"key words {sorted(key_words(expected))}; parsed {cat}: {g['parsed'].iloc[0]}")

    print("\n== expert records that cannot come from prompt words (not counted) ==")
    for expected, g in records[skip].groupby("expected"):
        print(f"  {NOT_FROM_PROMPT_WORDS[expected]}: {expected!r} ({', '.join(g['track_id'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()

    df = load_sunocaps()
    reqs = parse_dataframe(df)
    out = reqs.assign(params=reqs["params"].map(json.dumps))
    out.to_csv(PROCESSED_DIR / "requirements.csv", index=False)
    print(f"saved {len(reqs)} requirements for {reqs['track_id'].nunique()} tracks "
          f"to {PROCESSED_DIR / 'requirements.csv'}\n")

    print_summary(df, reqs)
    if args.examples:
        print_examples(df, reqs)
    print_coverage(df, reqs, pd.read_csv(PROCESSED_DIR / "mismatch_records.csv"))


if __name__ == "__main__":
    main()
