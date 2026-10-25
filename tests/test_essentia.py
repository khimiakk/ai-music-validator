"""essentia integration in verdicts.py with synthetic scores."""
import numpy as np
import pandas as pd
import pytest

from aimv.verdicts import (TEMPO_COLS, add_essentia_voice, essentia_percentiles, essentia_track_scores,
                           load_essentia_scores, requirement_verdicts, system_inputs)

TRACKS = ["a_1", "b_1", "c_1", "d_1"]


@pytest.fixture
def essentia(tmp_path):
    """synthetic notebook-04 output with mean and max per label."""
    mean = {
        "genre__Rock---Indie Rock": [0.9, 0.1, 0.2, 0.3],
        "genre__Rock---Punk": [0.2, 0.8, 0.1, 0.1],
        "genre__Pop---Ballad": [0.1, 0.1, 0.7, 0.2],
        "instr__piano": [0.9, 0.2, 0.4, 0.1],
        "mood__happy": [0.3, 0.6, 0.1, 0.2],
        "voice__voice": [0.9, 0.1, 0.6, 0.2],
        "voice__instrumental": [0.1, 0.9, 0.4, 0.8],
    }
    df = pd.DataFrame({"track_id": TRACKS})
    for col, values in mean.items():
        df[f"{col}__mean"] = values
        df[f"{col}__max"] = np.minimum(np.array(values) + 0.1, 1)
    path = tmp_path / "essentia_scores.csv"
    df.to_csv(path, index=False)
    return load_essentia_scores(path)


def req(track, check, category, value, polarity=1, **params):
    return dict(req_id=f"{track}#00", track_id=track, original_id=track.split("_")[0], prompt_type="aspect_list",
                category=category, value=value, polarity=polarity, check=check, clap_text=value, params=params)


def test_missing_file_means_no_essentia(tmp_path):
    assert load_essentia_scores(tmp_path / "nope.csv") is None


def test_load_keeps_one_statistic(essentia, tmp_path):
    assert "instr__piano" in essentia.columns and len(essentia.columns) == 7
    assert essentia.loc["a_1", "instr__piano"] == 0.9
    maxed = load_essentia_scores(tmp_path / "essentia_scores.csv", stat="max")
    assert maxed.loc["a_1", "instr__piano"] == pytest.approx(1.0)


def test_track_scores_per_check(essentia):
    rock = essentia_track_scores(essentia, "essentia_genre", {"labels": ["Rock---"]})
    assert list(rock) == [0.9, 0.8, 0.2, 0.3]  # max over Indie Rock and Punk
    assert list(essentia_track_scores(essentia, "essentia_instrument", {"label": "piano"})) == [0.9, 0.2, 0.4, 0.1]
    assert essentia_track_scores(essentia, "essentia_genre", {"labels": ["Greek"]}) is None  # not in the models
    assert essentia_track_scores(essentia, "essentia_mood", {"label": "epic"}) is None


def test_percentiles_against_tracks_without_the_requirement(essentia):
    reqs = pd.DataFrame([
        req("a_1", "essentia_instrument", "instruments", "piano", label="piano"),
        req("c_1", "essentia_instrument", "instruments", "piano", label="piano"),
        req("a_1", "essentia_genre", "genre", "greek", labels=["Greek"]),
        req("e_1", "essentia_mood", "mood", "happy", label="happy"),  # track without Essentia scores
    ])
    out = essentia_percentiles(reqs, essentia, TRACKS + ["e_1"])
    # reference is b_1 and d_1; a and c beat both
    assert out.loc[0, "score_pct"] == 1.0 and out.loc[1, "score_pct"] == 1.0
    assert out.loc[0, "s"] == 0.9
    assert np.isnan(out.loc[2, "score_pct"])  # label not in the models -> keeps CLAP
    assert np.isnan(out.loc[3, "score_pct"])  # no Essentia scores for this track -> keeps CLAP


def test_system_inputs(essentia):
    evidence = pd.DataFrame({"p_vocals": [0.2, 0.8], "p_female": 0.5}, index=pd.Index(["a_1", "b_1"], name="track_id"))
    ev = add_essentia_voice(evidence, essentia)
    assert list(ev["p_voice"]) == [0.9, 0.1]
    clap = pd.DataFrame({"s": [0.3, 0.4], "score_pct": [0.2, 0.6]}, index=[0, 1])
    ess = pd.DataFrame({"s": [0.9, np.nan], "score_pct": [1.0, np.nan]}, index=[0, 1])  # row 1: no Essentia label

    _, scores = system_inputs("clap", ev, clap, ess)
    assert list(scores["score_pct"]) == [0.2, 0.6]
    e_ev, scores = system_inputs("essentia", ev, clap, ess)
    assert list(scores["score_pct"]) == [1.0, 0.6] and list(e_ev["p_vocals"]) == [0.9, 0.1]
    a_ev, scores = system_inputs("average", ev, clap, ess)
    assert list(scores["score_pct"]) == [0.6, 0.6] and list(a_ev["p_vocals"]) == pytest.approx([0.55, 0.45])
    _, scores = system_inputs("essentia", ev, clap, None)  # no Essentia file -> CLAP
    assert list(scores["score_pct"]) == [0.2, 0.6]
    with pytest.raises(ValueError):
        system_inputs("mulan", ev, clap, ess)


def test_essentia_voice_drives_the_vocal_verdict(essentia):
    reqs = pd.DataFrame([req("a_1", "vocal_presence", "vocals", "vocals", polarity=-1)])  # prompt says instrumental
    evidence = pd.DataFrame({"p_vocals": [0.2], "p_female": [0.5]}, index=pd.Index(["a_1"], name="track_id"))
    evidence[TEMPO_COLS] = 1 / 3
    ev = add_essentia_voice(evidence, essentia)
    clap_ev, _ = system_inputs("clap", ev, pd.DataFrame(columns=["s", "score_pct"]), None)
    ess_ev, _ = system_inputs("essentia", ev, pd.DataFrame(columns=["s", "score_pct"]),
                              pd.DataFrame(columns=["s", "score_pct"]))
    empty = pd.DataFrame()
    assert requirement_verdicts(reqs, clap_ev, empty, [1 / 3] * 3).loc[0, "verdict"] == "satisfied"  # p_vocals 0.2 from CLAP
    assert requirement_verdicts(reqs, ess_ev, empty, [1 / 3] * 3).loc[0, "verdict"] == "ignored"  # p_voice 0.9 from Essentia
