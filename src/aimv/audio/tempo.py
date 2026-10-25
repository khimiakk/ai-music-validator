"""tempo estimation and the tempo checks."""
from __future__ import annotations

import numpy as np

TEMPO_SR = 22_050

TEMPO_CLASSES = ["slow", "medium", "fast"]

# tempo word -> tempo classes that satisfy it
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
    # beat_regularity: evenness of beat intervals, not a real confidence
    conf = float(max(0.0, 1.0 - ibi.std() / ibi.mean())) if len(ibi) > 2 and ibi.mean() > 0 else 0.0
    return {"bpm": bpm, "beat_regularity": conf, "n_beats": int(len(beats))}


def estimate_tempo_flat_prior(y: np.ndarray, sr: int, max_seconds: float = 120.0) -> float:
    """librosa tempo with a flat prior over 40-220 BPM."""
    import librosa
    from scipy import stats

    onset_env = _onset_envelope(y, sr, max_seconds)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=TEMPO_SR, prior=stats.uniform(40, 180))
    return float(tempo[0])


def prior_correct(probs, class_means) -> np.ndarray:
    """divide class probabilities by their means, then renormalise."""
    p = np.asarray(probs, dtype=float) / np.asarray(class_means, dtype=float)
    return p / p.sum()


def tempo_word_verdict(probs, word: str, class_means=None) -> tuple[str, str]:
    """verdict for a tempo word from (prior-corrected) CLAP class probabilities."""
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
    """verdict for an explicit BPM requirement, with octave tolerance."""
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
