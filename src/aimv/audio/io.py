"""Audio loading and windowed analysis for long tracks."""
from __future__ import annotations

import numpy as np

CLAP_SR = 48_000


def load_audio(path, sr: int = CLAP_SR) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def window_starts(n_samples: int, sr: int, win_s: float = 10.0, n_windows: int | None = 3,
                  hop_s: float | None = None) -> list[int]:
    """Start indices of analysis windows.
    """
    win = int(win_s * sr)
    if n_samples <= win:
        return [0]
    last = n_samples - win
    if n_windows is None:
        hop = int((hop_s or win_s) * sr)
        return list(range(0, last + 1, hop))
    if n_windows == 1:
        return [last // 2]
    return [int(round(i * last / (n_windows - 1))) for i in range(n_windows)]


def windows(y: np.ndarray, sr: int, win_s: float = 10.0, n_windows: int | None = 3,
            hop_s: float | None = None) -> list[np.ndarray]:
    win = int(win_s * sr)
    out = []
    for s in window_starts(len(y), sr, win_s, n_windows, hop_s):
        w = y[s:s + win]
        if len(w) < win:
            reps = int(np.ceil(win / max(len(w), 1)))
            w = np.tile(w, reps)[:win]
        out.append(w)
    return out
