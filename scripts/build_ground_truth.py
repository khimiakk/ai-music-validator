"""Stage 2: expert comments -> mismatch records -> ground truth (no audio needed).

    build_ground_truth.py            # build, save, print a summary
    build_ground_truth.py --review   # also print the comments that still need a manual look

    data/processed/mismatch_records.csv           one row per (track, extracted issue)
    data/processed/ground_truth_per_category.csv  one row per (track, category)
    data/processed/ground_truth_per_track.csv     one row per track
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimv.annotations import (COMMENT_RULES, RULES_V2, build_ground_truth,  # noqa: E402
                              build_mismatch_records, rule_v2_records)
from aimv.data import PROCESSED_DIR, has_text, load_sunocaps  # noqa: E402

# notebook 02, section 2.5: the extended rule extractor (RULES_V2)
NOTEBOOK_RULE_V2 = {"records": 117, "tracks": 104}
REVIEW_SOURCES = ["manual_draft_03", "generic_no_instrument"]
NEEDS_REVIEW = ["needs_review", "needs_audio_review"]


def print_summary(df, records, per_category, per_track):
    commented = per_track[per_track["has_comment"]]

    print("== records by source x review_status ==")
    print(pd.crosstab(records["source"], records["review_status"], margins=True).to_string())

    # the notebook-02 extractor alone (manual review 04 relabels some of its records afterwards)
    n_rule_v2 = df.loc[has_text(df["Comments"]), "Comments"].map(lambda c: len(rule_v2_records(c.lower().strip())))
    got = {"records": int(n_rule_v2.sum()), "tracks": int((n_rule_v2 > 0).sum())}
    status = "OK" if got == NOTEBOOK_RULE_V2 else "DIFFERENT"
    relabelled = records["review_notes"].str.startswith("extracted by rule_v2", na=False).sum()
    print(f"\nRULES_V2 extractor vs notebook 02: {got['records']} records on {got['tracks']} tracks "
          f"(notebook: {NOTEBOOK_RULE_V2['records']} on {NOTEBOOK_RULE_V2['tracks']}) -> {status}")
    print(f"  in the final records: {(records['source'] == 'rule_v2').sum()} with source rule_v2 "
          f"+ {relabelled} relabelled by manual_review_04 (error_type incorrect -> added)")

    still = records["review_status"].value_counts().reindex(NEEDS_REVIEW, fill_value=0)
    print(f"\nrecords that still need review: {still.sum()} of {len(records)} ({still.to_dict()})")

    print(f"\n== comment_type over the {len(commented)} commented tracks ==")
    print(commented["comment_type"].value_counts().to_string())

    print("\n== ground-truth mismatch rate per category x prompt_type ==")
    rate = per_category.pivot_table(index="category", columns="prompt_type", values="gt_mismatch", aggfunc="mean")
    rate["all"] = per_category.groupby("category")["gt_mismatch"].mean()
    print(rate.round(3).to_string())

    print("\n== verdict counts per category (all records) ==")
    print(pd.crosstab(records["category"], records["verdict"], margins=True).to_string())

    print("\n== vocals_gt_source ==")
    print(per_track["vocals_gt_source"].value_counts().to_string())


def print_review(df, records, per_track):
    print("\n\n== records from draft rules / generic 'No <instrument>' (please check) ==")
    review = records[records["source"].isin(REVIEW_SOURCES)]
    for comment, group in review.groupby("expert_comment", sort=False):
        print(f"\nCOMMENT ({group['track_id'].nunique()} tracks: {', '.join(group['track_id'].unique())}):")
        print(f"  {comment!r}")
        for _, r in group.drop_duplicates(["category", "expected", "observed", "error_type"]).iterrows():
            print(f"  -> [{r['source']}] {r['category']}: {r['expected']} -> {r['observed']} "
                  f"({r['error_type']} / {r['verdict']}), evidence {r['evidence']!r}")

    print("\n\n== comments with comment_type 'unparsed' ==")
    unparsed = per_track[per_track["comment_type"] == "unparsed"].merge(df[["track_id", "Comments"]], on="track_id")
    for comment, group in unparsed.groupby("Comments", sort=False):
        print(f"\n({len(group)} tracks: {', '.join(group['track_id'])})\n  {comment!r}")

    print("\n\n== how many distinct comments each rule matches (0 = rule never fires) ==")
    comments = df.loc[has_text(df["Comments"]), "Comments"].str.lower().str.strip().unique()
    rules = [("rule_v2", r["pattern"]) for r in RULES_V2] + [(cr.source, cr.pattern) for cr in COMMENT_RULES]
    for source, pattern in rules:
        n = sum(bool(re.search(pattern, c)) for c in comments)
        print(f"  {n:3d}  {source:<16} {pattern}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()

    df = load_sunocaps()
    records = build_mismatch_records(df)
    per_category, per_track = build_ground_truth(df, records)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records.to_csv(PROCESSED_DIR / "mismatch_records.csv", index=False)
    per_category.to_csv(PROCESSED_DIR / "ground_truth_per_category.csv", index=False)
    per_track.to_csv(PROCESSED_DIR / "ground_truth_per_track.csv", index=False)
    print(f"saved {len(records)} records, {len(per_category)} track x category rows, "
          f"{len(per_track)} tracks to {PROCESSED_DIR}\n")

    print_summary(df, records, per_category, per_track)
    if args.review:
        print_review(df, records, per_track)


if __name__ == "__main__":
    main()
