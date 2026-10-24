"""verdicts.py with synthetic embeddings (no CLAP model needed)."""
import numpy as np
import pandas as pd
import pytest

from aimv.verdicts import (TEMPO_COLS, apply_tau, clap_percentiles, clap_verdict, hallucination_rows,
                           predictions_per_category, requirement_verdicts, softmax_probs)


def unit(*v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# 2-d "embedding space": the first axis means "piano"
TEXTS = {"piano melody": unit(1, 0), "music with piano": unit(1, 0)}
TRACKS = ["a_1", "b_1", "c_1", "d_1"]
E = np.stack([unit(1, 0.1), unit(0.9, 0.5), unit(0.2, 1), unit(-0.5, 1)])  # a sounds most like piano


def embed_text(texts):
    return np.stack([TEXTS[t] for t in texts])


def req(track, check="clap", polarity=1, category="instruments", value="piano", clap_text="piano melody", **params):
    return dict(req_id=f"{track}#00", track_id=track, original_id=track.split("_")[0], prompt_type="aspect_list",
                category=category, value=value, polarity=polarity, check=check, clap_text=clap_text, params=params)


def evidence(**tracks):
    """evidence(a_1=(p_vocals, p_female)) with flat tempo probabilities."""
    ev = pd.DataFrame({t: dict(p_vocals=v, p_female=f) for t, (v, f) in tracks.items()}).T
    ev[TEMPO_COLS] = 1 / 3
    return ev.rename_axis("track_id")


def test_softmax_probs():
    p = softmax_probs(E, embed_text(["piano melody"] * 2) * [[1], [-1]], logit_scale=10)
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0, 0] > 0.99 and p[3, 0] < 0.5  # track a is close to "piano", d is not


def test_percentile_against_tracks_without_the_requirement():
    reqs = pd.DataFrame([req("a_1"), req("c_1")])
    scores = clap_percentiles(reqs, TRACKS, E, embed_text)
    # reference = tracks b and d (no piano requirement): sims 0.87 and -0.45
    assert scores.loc[0, "score_pct"] == 1.0  # a (0.995) beats both
    assert scores.loc[1, "score_pct"] == 0.5  # c (0.20) beats only d


@pytest.mark.parametrize("pct, polarity, expected", [
    (0.9, 1, "satisfied"), (0.2, 1, "ignored"), (0.9, -1, "ignored"), (0.2, -1, "satisfied")])
def test_clap_verdict(pct, polarity, expected):
    assert clap_verdict(pct, polarity, tau=0.5) == expected


def test_prohibition_scores_the_element_itself():
    # "no piano": CLAP scores "music with piano" and the verdict is reversed
    reqs = pd.DataFrame([req("a_1", polarity=-1, clap_text="music without piano")])
    scores = clap_percentiles(reqs, TRACKS, E, embed_text)
    assert scores.loc[0, "text"] == "music with piano"
    v = requirement_verdicts(reqs, evidence(a_1=(0.5, 0.5)), scores, class_means=[1 / 3] * 3)
    assert v.loc[0, "verdict"] == "ignored" and v.loc[0, "mismatch_score"] == 1.0


def test_vocal_checks():
    reqs = pd.DataFrame([
        req("a_1", "vocal_presence", 1, "vocals", "vocals"),
        req("a_1", "vocal_presence", -1, "vocals", "vocals"),
        req("a_1", "vocal_gender", 1, "vocals", "female vocals", gender="female"),
        req("a_1", "vocal_gender", 1, "vocals", "male vocals", gender="male"),
        req("b_1", "vocal_presence", 1, "vocals", "vocals"),
        req("b_1", "vocal_gender", 1, "vocals", "female vocals", gender="female"),
    ])
    ev = evidence(a_1=(0.9, 0.2), b_1=(0.1, 0.9))  # a: vocals, sounds male; b: no vocals
    v = requirement_verdicts(reqs, ev, pd.DataFrame(), class_means=[1 / 3] * 3)
    assert list(v["verdict"]) == ["satisfied", "ignored", "changed", "satisfied", "ignored", "not_checked"]
    assert v.loc[0, "mismatch_score"] == pytest.approx(0.1)  # 1 - p_vocals
    assert v.loc[2, "mismatch_score"] == pytest.approx(0.8)  # 1 - p(female)


def test_tempo_uses_prior_correction():
    reqs = pd.DataFrame([req("a_1", "tempo", 1, "tempo", "fast")])
    ev = evidence(a_1=(0.5, 0.5))
    ev.loc["a_1", TEMPO_COLS] = [0.5, 0.2, 0.3]  # raw argmax = slow
    v = requirement_verdicts(reqs, ev, pd.DataFrame(), class_means=[0.6, 0.2, 0.2])  # corrected argmax = fast
    assert v.loc[0, "verdict"] == "satisfied" and v.loc[0, "mismatch_score"] == 0.0


def test_hallucination_only_without_a_vocals_requirement():
    reqs = pd.DataFrame([req("a_1", "vocal_presence", 1, "vocals", "vocals")])
    rows = hallucination_rows(reqs, evidence(a_1=(0.9, 0.5), b_1=(0.8, 0.5), c_1=(0.3, 0.5))).set_index("track_id")
    assert "a_1" not in rows.index  # the prompt asks for vocals
    assert rows.loc["b_1", "verdict"] == "hallucinated" and rows.loc["c_1", "verdict"] == "none"
    assert rows.loc["b_1", "category"] == "vocals" and rows.loc["b_1", "prompt_type"] == "aspect_list"


def test_predictions_per_category():
    verdicts = pd.DataFrame([
        dict(track_id="a_1", category="instruments", check="clap", verdict="satisfied", mismatch_score=0.2),
        dict(track_id="a_1", category="instruments", check="clap", verdict="ignored", mismatch_score=0.7),
        dict(track_id="a_1", category="vocals", check="vocal_hallucination", verdict="none", mismatch_score=0.3),
    ])
    p = predictions_per_category(verdicts, ["a_1"]).set_index("category")
    assert len(p) == 6  # every category, also those the prompt does not mention
    assert p.loc["instruments", "pred_mismatch"] and p.loc["instruments", "pred_score"] == 0.7
    assert p.loc["instruments", "n_requirements"] == 2
    assert not p.loc["vocals", "pred_mismatch"] and p.loc["vocals", "n_requirements"] == 0
    assert not p.loc["genre", "pred_mismatch"] and p.loc["genre", "pred_score"] == 0


def test_apply_tau_changes_only_clap_checks():
    verdicts = pd.DataFrame([
        dict(check="clap", polarity=1, score_pct=0.6, verdict="satisfied"),
        dict(check="vocal_presence", polarity=1, score_pct=np.nan, verdict="ignored"),
    ])
    v = apply_tau(verdicts, tau=0.7)
    assert list(v["verdict"]) == ["ignored", "ignored"]
    assert list(apply_tau(verdicts, tau=0.5)["verdict"]) == ["satisfied", "ignored"]
