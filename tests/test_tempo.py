"""tempo estimation on synthetic click tracks and the tempo checks."""
import librosa
import pytest

from aimv.audio.tempo import estimate_tempo, prior_correct, tempo_verdict, tempo_word_verdict

SR = 48_000  # same rate the extraction script uses (CLAP rate)


def click_track(bpm: float, seconds: float = 30.0, sr: int = SR):
    times = [i * 60.0 / bpm for i in range(int(seconds * bpm / 60))]
    return librosa.clicks(times=times, sr=sr, length=int(seconds * sr))


@pytest.mark.parametrize("bpm", [90, 120, 140])
def test_estimate_tempo_click_track(bpm):
    est = estimate_tempo(click_track(bpm), SR)["bpm"]
    # accept half or double tempo, beat trackers often lock there
    ok = any(abs(est - bpm * f) <= 0.05 * bpm * f for f in (1.0, 0.5, 2.0))
    assert ok, f"estimated {est:.1f} bpm for a {bpm} bpm click track"


def test_tempo_verdict_inside_range():
    assert tempo_verdict(130, 120, 200)[0] == "satisfied"


def test_tempo_verdict_outside_range():
    assert tempo_verdict(100, 120, 200)[0] == "changed"


def test_tempo_verdict_octave():
    verdict, detail = tempo_verdict(60, 118, 122)
    assert verdict == "satisfied"
    assert detail == "octave_double"


# probs are CLAP probabilities for (slow, medium, fast)
@pytest.mark.parametrize("probs, word, expected", [
    ([0.1, 0.2, 0.7], "fast", "satisfied"),
    ([0.1, 0.2, 0.7], "slow", "changed"),
    ([0.6, 0.3, 0.1], "slow", "satisfied"),
    ([0.2, 0.5, 0.3], "medium fast", "satisfied"),
    ([0.2, 0.5, 0.3], "medium-fast", "satisfied"),
    ([0.2, 0.5, 0.3], "downtempo", "satisfied"),
    ([0.1, 0.2, 0.7], "downtempo", "changed"),
    ([0.2, 0.5, 0.3], "uptempo", "changed"),
    ([0.2, 0.5, 0.3], "mid", "satisfied"),  # parsed from "mid tempo"
    ([0.1, 0.2, 0.7], "faster", "satisfied"),  # parsed from "faster tempo"
    ([0.2, 0.5, 0.3], "groovy", "unknown"),
])
def test_tempo_word_verdict(probs, word, expected):
    assert tempo_word_verdict(probs, word)[0] == expected


def test_prior_correct():
    p = prior_correct([0.5, 0.2, 0.3], [0.6, 0.2, 0.2])
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 2  # ratios 0.5/0.6 < 0.2/0.2 < 0.3/0.2, so fast wins
    assert prior_correct([0.5, 0.2, 0.3], [1 / 3] * 3) == pytest.approx([0.5, 0.2, 0.3])


def test_tempo_word_verdict_prior_corrected():
    probs, means = [0.5, 0.2, 0.3], [0.6, 0.2, 0.2]  # raw argmax = slow, corrected argmax = fast
    assert tempo_word_verdict(probs, "slow")[0] == "satisfied"
    assert tempo_word_verdict(probs, "slow", means)[0] == "changed"
    assert tempo_word_verdict(probs, "fast", means)[0] == "satisfied"
