"""Tempo estimation and a tolerant tempo check (Mustango-style).
"""
from __future__ import annotations

import numpy as np

TEMPO_SR = 22_050


def estimate_tempo(y: np.ndarray, sr: int, max_seconds: float = 120.0) -> dict:
    import librosa

    y = y[: int(max_seconds * sr)]
    if sr != TEMPO_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TEMPO_SR)
        sr = TEMPO_SR
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    ibi = np.diff(librosa.frames_to_time(beats, sr=sr))
    conf = float(max(0.0, 1.0 - ibi.std() / ibi.mean())) if len(ibi) > 2 and ibi.mean() > 0 else 0.0
    return {"bpm": bpm, "beat_regularity": conf, "n_beats": int(len(beats))}


def tempo_verdict(bpm: float, bpm_min: float, bpm_max: float, tol: float = 0.08,
                  allow_octave: bool | None = None) -> tuple[str, str]:
    """Return (verdict, detail). verdict in {satisfied, changed, unknown}.
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
