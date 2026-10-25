"""annotations: comments to records and ground truth on the fixture."""
from pathlib import Path

import pytest

from aimv import CATEGORIES
from aimv.annotations import (MISMATCH_ERROR_TYPES, build_ground_truth, build_mismatch_records, extract_records,
                              rule_v2_records, vocals_from_comment)
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
    tid = "-r7iz-9v9bA_3"  # comment: follows very well the no-lyrics only vocables prompt
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
    records = records[records["error_type"].isin(MISMATCH_ERROR_TYPES)]  # partial records are not counted by default
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
    assert t.loc["-r7iz-9v9bA_4", "vocals_present_gt"] == 0.0  # comment says "does not have voice"
    assert t.loc["-r7iz-9v9bA_4", "vocals_gt_source"] == "comment"
    assert t.loc["1y1lEOGBcWM_1", "vocals_present_gt"] == 1.0  # comment says "added voice"
    assert t.loc["-r7iz-9v9bA_3", "vocals_present_gt"] == 1.0  # comment says "only vocables prompt" (review 04)
    assert t.loc["-r7iz-9v9bA_3", "vocals_gt_source"] == "comment"


def test_not_instrumental_plus_composition(data):
    _, records, _, _ = data
    r = records[records["track_id"] == "2ShO1jZYZeA_3"].set_index("category")
    assert r.loc["vocals", "error_type"] == "added" and r.loc["vocals", "verdict"] == "ignored"
    assert r.loc["other", "error_type"] == "incorrect" and r.loc["other", "observed"] == "not followed"


@pytest.mark.parametrize("comment, category, error_type", [
    ("No strings", "instruments", "missing"),
    ("Not instrumental.", "vocals", "added"),  # review 04: was incorrect in notebook 02
    ("Does not follow instrumental prompt.", "vocals", "added"),
    ("Could slightly pass as low quality due to few instruments, but still far way from low.", "other", "incorrect"),
    ("No sitar", "instruments", "missing"),  # generic "No <instrument>" rule
    ("No humming.", "vocals", "missing"),
    ("Labaled instrumental by the generator but has lyrics", "vocals", "unexpected_addition"),
    ("Very inapropiate as ads a foreign hindi accent", "vocals", "unexpected_addition"),
])
def test_extract_records_single(comment, category, error_type):
    records, _ = extract_records(comment)
    assert [(r["category"], r["error_type"]) for r in records] == [(category, error_type)]


def test_rule_v2_extractor_is_unchanged():
    # rule_v2 alone says "incorrect"; review 04 relabels it afterwards
    raw = rule_v2_records("not instrumental.")
    assert [(r["source"], r["error_type"]) for r in raw] == [("rule_v2", "incorrect")]
    (rec,), _ = extract_records("Not instrumental.")
    assert (rec["source"], rec["review_status"]) == ("manual_review_04", "text_reviewed")


@pytest.mark.parametrize("comment, expected, source", [
    ("Not clearly male voice", "male vocals", "manual_review_04"),
    ("Not clearly female vocals", "female vocals", "manual_review_04"),
    ("Not clearly a male singer", "male vocals", "rule_v2"),  # already a rule_v2 match, no duplicate
    ("Not clearly male voices.", "male vocals", "rule_v2"),
])
def test_unclear_voice(comment, expected, source):
    records, _ = extract_records(comment)
    assert [(r["expected"], r["error_type"], r["source"]) for r in records] == [(expected, "incorrect", source)]


def test_makes_it_instrumental():
    records, _ = extract_records("Does not follow the vocal command and makes it instrumental. Not following the "
                                 "piano and synth are doing the melody command and instead using a guitar.")
    labels = {(r["category"], r["expected"], r["observed"], r["error_type"]) for r in records}
    assert labels == {("vocals", "vocals", "absent", "missing"),
                      ("instruments", "piano and synth", "guitar", "replaced")}


def test_foreign_language_is_not_a_mismatch():
    comment = 'lyrics generated in Spanish in response to "foreign language" prompt'
    assert extract_records(comment) == ([], "descriptive")
    assert vocals_from_comment(comment) == 1.0


def test_review_04_records_are_text_reviewed():
    comments = ["Not instrumental.", "No humming.", "Not clearly male voice", "Nothing polish in it.",
                "Could slightly pass as low quality", "Specific instruction on musical terms about the "
                "composition are not followed (in which bar happens what)."]
    for comment in comments:
        records, _ = extract_records(comment)
        assert records and all(r["source"] == "manual_review_04" for r in records), comment
        assert all(r["review_status"] == "text_reviewed" for r in records), comment


def test_missing_comment():
    assert extract_records(None) == ([], "missing")
    assert extract_records("   ") == ([], "missing")
