"""AI Music Generation Validator (aimv).

Pipeline modules:
    data            – loading SunoCaps and derived columns
    annotations     – expert-comment rules -> mismatch records -> ground truth
    prompt_parsing  – prompt -> list of checkable requirements
    audio           – audio feature extraction (tempo, vocals, Essentia, CLAP)
    verdicts        – requirement + audio evidence -> verdicts
    evaluation      – metrics against expert ground truth
"""

CATEGORIES = ["genre", "instruments", "vocals", "tempo", "mood", "other"]

__all__ = ["CATEGORIES"]
