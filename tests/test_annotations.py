"""Expert comments -> mismatch records -> ground truth, on the hand-made fixture."""
from pathlib import Path

import pytest

from aimv import CATEGORIES
from aimv.annotations import MISMATCH_ERROR_TYPES, build_ground_truth, build_mismatch_records, extract_records
from aimv.data import load_sunocaps

FIXTURE = Path(__file__).parent / "fixtures" / "sunocaps_sample.csv"
LABEL_COLS = ["category", "expected", "observed", "error_type", "verdict"]


@pytest.fixture(scope="module")
def data():
    df = load_sunocaps(FIXTURE)
    records = build_mismatch_records(df)
    per_category, per_track = build_ground_truth(df, records)
    return df, records, per_category, per_track


def test_added_voice_becomes_hallucinated(data):
    _, records, _, _ = data
    vocals = records[records["track_id"].str.startswith("1y1lEOGBcWM") & (records["category"] == "vocals")]
    assert len(vocals) == 4  # all 4 versions share the comment
    assert (vocals["error_type"] == "unexpected_addition").all()
    assert (vocals["verdict"] == "hallucinated").all()
    assert (vocals["expected"] == "vocals not specified").all()


def test_positive_comment_gives_no_records(data):
    _, records, per_category, per_track = data
    tid = "-r7iz-9v9bA_3"  # "Follows very well the no-lyrics only vocables prompt"
    assert extract_records("Follows very well the no-lyrics only vocables prompt ") == ([], "positive")
    assert tid not in set(records["track_id"])
    track = per_track.set_index("track_id").loc[tid]
    assert track["comment_type"] == "positive" and track["any_mismatch"] == 0
    assert (per_category.loc[per_category["track_id"] == tid, "gt_source"] == "comment_no_issue").all()


def test_same_comment_same_labels(data):
    df, records, _, _ = data
    for comment, tracks in df.dropna(subset=["Comments"]).groupby("Comments")["track_id"]:
        if len(tracks) < 2:
            continue
        labels = [sorted(map(tuple, records.loc[records["track_id"] == t, LABEL_COLS].values)) for t in tracks]
        assert all(lab == labels[0] for lab in labels), comment


def test_ground_truth_shapes(data):
    df, _, per_category, per_track = data
    assert len(per_track) == len(df)
    assert len(per_category) == len(df) * len(CATEGORIES)
    assert set(per_category["category"]) == set(CATEGORIES)


def test_gt_mismatch_follows_records(data):
    _, records, per_category, _ = data
    records = records[records["error_type"].isin(MISMATCH_ERROR_TYPES)]  # "partial" is not counted by default
    has_record = set(zip(records["track_id"], records["category"]))
    for _, row in per_category.iterrows():
        assert row["gt_mismatch"] == int((row["track_id"], row["category"]) in has_record)


def test_no_comment_is_assumed_ok(data):
    _, _, per_category, per_track = data
    tid = "0hfU27A6tus_1"  # empty comment
    assert not per_track.set_index("track_id").loc[tid, "has_comment"]
    rows = per_category[per_category["track_id"] == tid]
    assert (rows["gt_source"] == "no_comment_assumed_ok").all() and (rows["gt_mismatch"] == 0).all()


def test_vocals_label_from_comment(data):
    _, _, _, per_track = data
    t = per_track.set_index("track_id")
    assert t.loc["-r7iz-9v9bA_4", "vocals_present_gt"] == 0.0  # "... Does not have voice"
    assert t.loc["-r7iz-9v9bA_4", "vocals_gt_source"] == "comment"
    assert t.loc["1y1lEOGBcWM_1", "vocals_present_gt"] == 1.0  # "Added voice ..."


@pytest.mark.parametrize("comment, category, error_type", [
    ("No strings", "instruments", "missing"),
    ("Not instrumental.", "vocals", "incorrect"),
    ("Could slightly pass as low quality due to few instruments", "other", "partial"),
    ("No sitar", "instruments", "missing"),  # generic "No <instrument>" rule
])
def test_extract_records_single(comment, category, error_type):
    records, _ = extract_records(comment)
    assert [(r["category"], r["error_type"]) for r in records] == [(category, error_type)]


def test_missing_comment():
    assert extract_records(None) == ([], "missing")
    assert extract_records("   ") == ([], "missing")
