"""loading SunoCaps and adding the derived columns."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "SunoCaps_dataset.csv"
AUDIO_DIR = DATA_DIR / "raw" / "audio"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"

PROMPT_TYPE_BY_VERSION = {"1": "aspect_list", "2": "aspect_list", "3": "caption", "4": "caption"}

EXTRA_EMOTION_FIXES = {"agressive": "aggressive"}


def has_text(series: pd.Series) -> pd.Series:
    """true where a cell holds a non-empty string (NaN-safe)."""
    return series.notna() & series.astype("string").str.strip().ne("").fillna(False)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """same derived columns as notebooks 01/02, in one place."""
    df = df.copy()
    parts = df["ytid_V"].str.rsplit("_", n=1)
    df["track_id"] = df["ytid_V"]
    df["original_id"] = parts.str[0]
    df["generation_version"] = parts.str[-1]
    df["prompt_type"] = df["generation_version"].map(PROMPT_TYPE_BY_VERSION)
    df["actual_prompt"] = df["aspect_list"].where(df["prompt_type"].eq("aspect_list"), df["caption"])
    df["actual_prompt"] = df["actual_prompt"].fillna("").astype(str)

    if "Emotion" in df:
        df["emotion_clean"] = (
            df["Emotion"].astype("string").str.lower().str.strip().str.split(",").str[0].str.strip()
        )
    if "Extra_emotion" in df:
        df["extra_emotion_list"] = df["Extra_emotion"].apply(_split_extra_emotions)
    return df


def _split_extra_emotions(value) -> list[str]:
    if pd.isna(value):
        return []
    out = []
    for part in str(value).lower().split(","):
        part = part.strip()
        if part:
            out.append(EXTRA_EMOTION_FIXES.get(part, part))
    return out


def load_sunocaps(path: str | Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_derived_columns(df)


def audio_path(track_id: str, audio_dir: str | Path = AUDIO_DIR) -> Path:
    """locate the audio file for a track."""
    audio_dir = Path(audio_dir)
    for ext in (".mp3", ".wav", ".flac", ".ogg"):
        p = audio_dir / f"{track_id}{ext}"
        if p.exists():
            return p
    matches = list(audio_dir.rglob(f"{track_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No audio for {track_id} under {audio_dir}")
