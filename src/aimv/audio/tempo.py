"""Tempo estimation and tempo checks.

* ``tempo_verdict``      explicit BPM in the prompt (e.g. "120 bpm"): Mustango-style
  check with +-8% tolerance; half/double tempo is accepted for these narrow ranges.
* ``tempo_word_verdict`` tempo words ("slow", "fast", ...): CLAP zero-shot over
  slow / medium / fast. The librosa beat tracker is pulled towards ~120 BPM and
  could not separate fast from slow prompts (ROC-AUC 0.47, CLAP 0.90; see
  scripts/compare_tempo.py). CLAP itself leans towards "slow", so the class
  probabilities are prior-corrected (``prior_correct``) before the argmax.
"""
from __future__ import annotations

import numpy as np

TEMPO_SR = 22_050

TEMPO_CLASSES = ["slow", "medium", "fast"]

# tempo word from the prompt (see vocab.TEMPO_WORDS) -> tempo classes that satisfy it
TEMPO_WORD_CLASSES = {
    "very slow": ["slow"], "slow": ["slow"], "slow tempo": ["slow"],
    "downtempo": ["slow", "medium"],
    "medium": ["medium"], "moderate": ["medium"], "mid-tempo": ["medium"], "mid": ["medium"],
    "medium-fast": ["medium", "fast"], "medium fast": ["medium", "fast"],
    "fast": ["fast"], "uptempo": ["fast"], "up-tempo": ["fast"], "quick": ["fast"], "very fast": ["fast"],
    "faster": ["fast"],
}


def _onset_envelope(y: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    import librosa

    y = y[: int(max_seconds * sr)]
    if sr != TEMPO_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TEMPO_SR)
    return librosa.onset.onset_strength(y=y, sr=TEMPO_SR)


def estimate_tempo(y: np.ndarray, sr: int, max_seconds: float = 120.0) -> dict:
    import librosa

    onset_env = _onset_envelope(y, sr, max_seconds)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=TEMPO_SR)
    bpm = float(np.atleast_1d(tempo)[0])
    ibi = np.diff(librosa.frames_to_time(beats, sr=TEMPO_SR))
    # beat_regularity = how even the beat intervals are; only a rough indicator, not a real confidence
    conf = float(max(0.0, 1.0 - ibi.std() / ibi.mean())) if len(ibi) > 2 and ibi.mean() > 0 else 0.0
    return {"bpm": bpm, "beat_regularity": conf, "n_beats": int(len(beats))}


def estimate_tempo_flat_prior(y: np.ndarray, sr: int, max_seconds: float = 120.0) -> float:
    """librosa tempo with a flat prior over 40-220 BPM (the default prior is centred on 120 BPM)."""
    import librosa
    from scipy import stats

    onset_env = _onset_envelope(y, sr, max_seconds)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=TEMPO_SR, prior=stats.uniform(40, 180))
    return float(tempo[0])


def prior_correct(probs, class_means) -> np.ndarray:
    """Divide each class probability by its mean over all tracks, then renormalise."""
    p = np.asarray(probs, dtype=float) / np.asarray(class_means, dtype=float)
    return p / p.sum()


def tempo_word_verdict(probs, word: str, class_means=None) -> tuple[str, str]:
    """Verdict for a tempo word. ``probs``: CLAP softmax over TEMPO_CLASSES (slow, medium, fast).

    ``class_means``: mean CLAP probability of each class over all 256 tracks. If given,
    ``probs`` are prior-corrected first (without it this is the plain argmax).
    satisfied if the top class is one of the classes the word accepts, otherwise changed.
    """
    if word not in TEMPO_WORD_CLASSES:
        return "unknown", f"unknown tempo word {word!r}"
    if class_means is not None:
        probs = prior_correct(probs, class_means)
    top = TEMPO_CLASSES[int(np.argmax(probs))]
    if top in TEMPO_WORD_CLASSES[word]:
        return "satisfied", f"CLAP top class {top}"
    return "changed", f"CLAP top class {top}, prompt asks {word}"


def tempo_verdict(bpm: float, bpm_min: float, bpm_max: float, tol: float = 0.08,
                  allow_octave: bool | None = None) -> tuple[str, str]:
    """Verdict for an explicit BPM requirement only (tempo words use tempo_word_verdict).

    Return (verdict, detail). verdict in {satisfied, changed, unknown}.
    """
    if bpm is None or not np.isfinite(bpm) or bpm <= 0:
        return "unknown", "no tempo estimate"
    if allow_octave is None:
        allow_octave = bpm_max / bpm_min <= 1.25
    lo, hi = bpm_min * (1 - tol), bpm_max * (1 + tol)
    if lo <= bpm <= hi:
        return "satisfied", "direct"
    if allow_octave:
        for factor, name in ((2.0, "double"), (0.5, "half")):
            if lo <= bpm * factor <= hi:
                return "satisfied", f"octave_{name}"
    return "changed", f"{bpm:.0f} bpm outside {bpm_min:.0f}-{bpm_max:.0f}"
