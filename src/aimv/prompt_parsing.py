"""Stage 3 – decompose a text prompt into checkable requirements.

Rule/vocabulary based (deterministic, no API needed). Works for both prompt
types:
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import vocab

# ---------------------------------------------------------------------------

@dataclass
class Requirement:
    track_id: str
    category: str          # genre | instruments | vocals | tempo | mood | other
    value: str             # normalised term, e.g. "electric guitar", "female vocals", "fast"
    phrase: str            # prompt phrase it came from
    polarity: int = 1      # +1 must be present, -1 must be absent
    check: str = "clap"    # tempo | vocal_presence | vocal_gender | essentia_* | clap
    clap_text: str = ""
    params: dict = field(default_factory=dict)
    source: str = ""       # aspect_list | caption

    @property
    def key(self):
        return (self.category, self.value, self.polarity)


def _alternation(terms):
    terms = sorted(set(terms), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(t) for t in terms) + r")(?:e?s)?(?![\w-])")


_RE = {
    "instruments": _alternation(vocab.INSTRUMENT_TERMS),
    "genre": _alternation(vocab.GENRE_TERMS),
    "mood": _alternation(vocab.MOOD_TERMS),
    "other": _alternation(vocab.OTHER_TERMS),
    "vocals": _alternation(vocab.VOCAL_TERMS),
    "tempo_word": _alternation(vocab.TEMPO_WORDS),
}
_BPM = re.compile(r"(\d{2,3})\s*(?:bpm|beats per minute)")
_NO_VOCALS = re.compile(r"\binstrumental\b(?!\s+(?:break|section|part|intro))|\b(?:no|without)\s+(?:vocals?|voices?|singers?|singing)\b|\bno one (?:is )?singing\b|\bthere (?:is|are) no (?:vocals?|singers?|singing)\b")
_NO_LYRICS = re.compile(r"\b(?:no|without)\s+(?:words|lyrics)\b|\bonly vocables\b")
_NO_INSTRUMENTS = re.compile(r"\b(?:no|without)\s+(?:instruments|instrumentation|accompaniment)\b|\ba ?cappella\b")
_FEMALE = re.compile(r"\b(?:female|woman|women|girl|girls|soprano|alto)\b")
_MALE = re.compile(r"\b(?:male|man|men|boy|boys|tenor|baritone)\b")


def split_phrases(text: str, prompt_type: str) -> list[str]:
    text = str(text or "").lower().strip()
    if not text:
        return []
    if prompt_type == "aspect_list":
        text = text.strip("[]")
        parts = re.split(r"\s*,\s*", text)
    else:
        parts = re.split(r"(?<=[.!?;])\s+|\.(?=[a-z])", text)
    return [p.strip(" '\".") for p in parts if p.strip(" '\".")]


def _covered(span, spans):
    return any(s <= span[0] and span[1] <= e for s, e in spans)


def parse_phrase(phrase: str, track_id: str = "", source: str = "") -> list[Requirement]:
    reqs: list[Requirement] = []
    neg_spans = []

    def add(**kw):
        reqs.append(Requirement(track_id=track_id, phrase=phrase, source=source, **kw))

    # ---- negations -------------------------------------------------------
    m = _NO_VOCALS.search(phrase)
    if m:
        neg_spans.append(m.span())
        add(category="vocals", value="vocals", polarity=-1, check="vocal_presence",
            clap_text="instrumental music without vocals")
    m = _NO_LYRICS.search(phrase)
    if m:
        neg_spans.append(m.span())
        add(category="vocals", value="lyrics", polarity=-1, check="clap",
            clap_text="singing vocables without words")
    m = _NO_INSTRUMENTS.search(phrase)
    if m:
        neg_spans.append(m.span())
        add(category="instruments", value="instruments", polarity=-1, check="clap",
            clap_text="a cappella singing without instruments")

    # ---- tempo -----------------------------------------------------------
    m = _BPM.search(phrase)
    if m:
        bpm = float(m.group(1))
        add(category="tempo", value=f"{int(bpm)} bpm", check="tempo",
            params=dict(bpm_min=bpm * 0.95, bpm_max=bpm * 1.05), clap_text=f"music at {int(bpm)} bpm")
    elif "tempo" in phrase or source == "aspect_list" or re.search(r"\bpaced?\b|\bbeat\b", phrase):
        for m in _RE["tempo_word"].finditer(phrase):
            word = m.group(1)
            if word == "medium" and "tempo" not in phrase:
                continue
            lo, hi = vocab.TEMPO_WORDS[word]
            label = word.replace(" tempo", "")
            add(category="tempo", value=label, check="tempo",
                params=dict(bpm_min=lo, bpm_max=hi), clap_text=f"{label} tempo music")
            break

    vocal_hits = [m for m in _RE["vocals"].finditer(phrase) if not _covered(m.span(), neg_spans)]
    if vocal_hits:
        add(category="vocals", value="vocals", check="vocal_presence", clap_text="a song with vocals")
        if _FEMALE.search(phrase):
            add(category="vocals", value="female vocals", check="vocal_gender",
                params=dict(gender="female"), clap_text="a song sung by a female vocalist")
        elif _MALE.search(phrase):
            add(category="vocals", value="male vocals", check="vocal_gender",
                params=dict(gender="male"), clap_text="a song sung by a male vocalist")
        for m in vocal_hits:
            term = m.group(1)
            if term in {"choir", "choral", "rap", "rapping", "rapper", "beatbox", "humming", "whistling",
                        "chant", "chanting", "falsetto", "spoken word", "yodel", "opera singer", "vocables",
                        "a cappella", "acapella", "harmonies", "backing vocals"}:
                add(category="vocals", value=term, check="clap", clap_text=phrase if source == "aspect_list" else term)

    for cat in ("instruments", "genre", "mood", "other"):
        for m in _RE[cat].finditer(phrase):
            if _covered(m.span(), neg_spans):
                continue
            term = m.group(1)
            params, check = {}, "clap"
            if cat == "instruments":
                if term in ("beat", "beats") and source == "caption":
                    continue
                if term in vocab.ESSENTIA_INSTRUMENT_MAP:
                    check, params = "essentia_instrument", dict(label=vocab.ESSENTIA_INSTRUMENT_MAP[term])
                clap = f"music with {term}"
            elif cat == "genre":
                if term in vocab.DISCOGS_GENRE_MAP:
                    check, params = "essentia_genre", dict(labels=vocab.DISCOGS_GENRE_MAP[term])
                clap = f"{term} music"
            elif cat == "mood":
                if term in vocab.ESSENTIA_MOOD_MAP:
                    check, params = "essentia_mood", dict(label=vocab.ESSENTIA_MOOD_MAP[term])
                clap = f"{term} music"
            else:
                if term == "live" and "live performance" in phrase:
                    continue
                clap = term if term not in ("noise", "noisy", "mono", "live") else f"{term} recording"
            if source == "aspect_list" and len(phrase) <= 60 and phrase != term:
                clap = phrase
            add(category=cat, value=term, check=check, params=params, clap_text=clap)

    # Aspect tags that matched nothing are still requirements (checked by CLAP)
    if not reqs and source == "aspect_list":
        add(category="other", value=phrase, check="clap", clap_text=phrase)
    return reqs


def parse_prompt(text: str, prompt_type: str, track_id: str = "") -> list[Requirement]:
    reqs, seen = [], set()
    for phrase in split_phrases(text, prompt_type):
        for r in parse_phrase(phrase, track_id=track_id, source=prompt_type):
            if r.key in seen:
                continue
            seen.add(r.key)
            reqs.append(r)
    if any(r.category == "vocals" and r.value == "vocals" and r.polarity == -1 for r in reqs):
        reqs = [r for r in reqs if not (r.category == "vocals" and r.polarity == 1 and r.check != "clap")]
    return reqs


def parse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Requirements for every track (uses ``actual_prompt`` and ``prompt_type``)."""
    rows = []
    for _, row in df.iterrows():
        for i, r in enumerate(parse_prompt(row["actual_prompt"], row["prompt_type"], row["track_id"])):
            d = asdict(r)
            d["req_id"] = f"{row['track_id']}#{i:02d}"
            d["original_id"] = row["original_id"]
            d["prompt_type"] = row["prompt_type"]
            rows.append(d)
    out = pd.DataFrame(rows)
    cols = ["req_id", "track_id", "original_id", "prompt_type", "category", "value", "polarity", "check",
            "clap_text", "params", "phrase", "source"]
    return out[cols] if len(out) else pd.DataFrame(columns=cols)
