"""Tempo estimation on synthetic click tracks + the tempo checks."""
import librosa
import pytest

from aimv.audio.tempo import estimate_tempo, tempo_verdict, tempo_word_verdict

SR = 48_000  # same rate the extraction script passes in (CLAP rate)


def click_track(bpm: float, seconds: float = 30.0, sr: int = SR):
    times = [i * 60.0 / bpm for i in range(int(seconds * bpm / 60))]
    return librosa.clicks(times=times, sr=sr, length=int(seconds * sr))


@pytest.mark.parametrize("bpm", [90, 120, 140])
def test_estimate_tempo_click_track(bpm):
    est = estimate_tempo(click_track(bpm), SR)["bpm"]
    # beat trackers often lock to half or double tempo, so accept those too
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
    ([0.2, 0.5, 0.3], "uptempo", "changed"),
    ([0.2, 0.5, 0.3], "groovy", "unknown"),
])
def test_tempo_word_verdict(probs, word, expected):
    assert tempo_word_verdict(probs, word)[0] == expected
