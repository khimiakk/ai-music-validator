"""expert comments to mismatch records and per-track ground truth."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import CATEGORIES
from .data import has_text


def _rule(pattern, category, expected, observed, error_type):
    return dict(pattern=pattern, category=category, expected=expected,
                observed=observed, error_type=error_type)


# rules from notebook 02, kept verbatim for reproducibility
MISMATCH_RULES = [
    _rule(r"\bno strings\b", "instruments", "strings", "absent", "missing"),
    _rule(r"\bno xylophone\b", "instruments", "xylophone", "absent", "missing"),
    _rule(r"\bno harmonica\b", "instruments", "harmonica", "absent", "missing"),
    _rule(r"\bno (?:digeridoo|didgeridoo)\b", "instruments", "didgeridoo", "absent", "missing"),
    _rule(r"\badded voice\b", "vocals", "no vocals", "vocals present", "added"),
    _rule(r"\bnot clearly a male (?:voice|vocal|singer)\b", "vocals", "male vocals", "unclear vocal type", "incorrect"),
    _rule(r"\bnot clearly a female (?:voice|vocal|singer)\b", "vocals", "female vocals", "unclear vocal type", "incorrect"),
    _rule(r"\bnot (?:low|poor) (?:audio )?quality\b", "other", "low recording quality", "higher recording quality", "incorrect"),
    _rule(r"\bno ambient noises?\b", "other", "ambient noise", "absent", "missing"),
]

ADDITIONAL_MISMATCH_RULES = [
    _rule(r"does not follow [\"']?instrumental", "vocals", "instrumental only", "instrumental requirement violated", "incorrect"),
    _rule(r"\bnot instrumental\b", "vocals", "instrumental only", "non-instrumental audio", "incorrect"),
    _rule(r"\bnot clearly female voice\b", "vocals", "female vocals", "unclear vocal type", "incorrect"),
    _rule(r"\bnot clearly male voices\b", "vocals", "male vocals", "unclear vocal type", "incorrect"),
    _rule(r"\bno irish bouzouki\b", "instruments", "irish bouzouki", "absent", "missing"),
    _rule(r"\bno bansuri flute\b", "instruments", "bansuri flute", "absent", "missing"),
    _rule(r"\bno woodwinds or violins\b", "instruments", "woodwinds and violins", "absent", "missing"),
    _rule(r"\bno sound of water\b", "other", "water sounds", "absent", "missing"),
    _rule(r"\bno ducks\b", "other", "duck sounds", "absent", "missing"),
    _rule(r"\bno glass shattering\b", "other", "glass shattering", "absent", "missing"),
]

RULES_V2 = MISMATCH_RULES + ADDITIONAL_MISMATCH_RULES

# review 04: these mean "added voice", so added, not incorrect
RULE_V2_FIXES = {
    r"does not follow [\"']?instrumental": dict(error_type="added"),
    r"\bnot instrumental\b": dict(error_type="added"),
}

# notebook 02, 2.8: vocals not prohibited here, so unexpected addition
UNEXPECTED_VOCALS_ORIGINAL_IDS = {"1y1lEOGBcWM", "298Q_pSLflU"}


@dataclass
class CommentRule:
    """annotation attached to every comment that matches ``pattern``."""
    pattern: str
    comment_type: str  # mismatch | positive | mixed | descriptive
    records: list[dict] = field(default_factory=list)
    review_status: str = "needs_review"
    source: str = "manual_review"
    notes: str = ""


def _rec(category, expected, observed, error_type):
    return dict(category=category, expected=expected, observed=observed, error_type=error_type)


COMMENT_RULES: list[CommentRule] = [
    # batch 1: notebook 02, section 2.10.1
    CommentRule(r"far from classical indian", "mixed",
                [_rec("genre", "Indian classical style", "closer to pop than classical Indian", "incorrect")],
                "needs_review", "manual_batch_01",
                "Classical Indian style is not explicitly specified in the aspect-list prompt."),
    CommentRule(r"does not follow [\"']?low quality[\"']? text command", "mismatch",
                [_rec("other", "low recording quality", "requested quality not achieved", "incorrect")],
                "text_reviewed", "manual_batch_01"),
    CommentRule(r"\bor nature sounds\b", "mismatch",
                [_rec("other", "nature sounds", "absent", "missing")],
                "text_reviewed", "manual_batch_01"),
    CommentRule(r"\bno noise or people talking\b", "mismatch",
                [_rec("other", "background chatter/noise", "absent", "missing")],
                "text_reviewed", "manual_batch_01"),
    # batch 2: notebook 02, section 2.10.2
    CommentRule(r"\bnot clearly a live performance\b", "mismatch",
                [_rec("other", "live performance", "not clearly live", "incorrect")],
                "text_reviewed", "manual_batch_02"),
    CommentRule(r"no clear noise as opposed to text comm", "mixed",
                [_rec("other", "noisy recording", "no clear noise", "missing")],
                "text_reviewed", "manual_batch_02",
                "Live character judged acceptable ('could be considered a live version')."),
    CommentRule(r"follows very well the no-lyrics only vocables", "positive", [],
                "text_reviewed", "manual_batch_02"),
    CommentRule(r"violins do not follow the eight(?:h)? note pattern", "mismatch",
                [_rec("other", "violins in eighth-note pattern", "mostly sixteenth-note pattern", "incorrect")],
                "text_reviewed", "manual_batch_02",
                "Rhythmic figure, not tempo: not checkable by BPM; kept in 'other'."),
    CommentRule(r"\bdoes not have (?:a )?voice\b", "mismatch",
                [_rec("vocals", "female vocals (vocables)", "absent", "missing")],
                "text_reviewed", "manual_batch_02"),
    CommentRule(r"instruments not specified are added", "mismatch",
                [_rec("instruments", "only the specified instruments", "additional instruments", "unexpected_addition")],
                "text_reviewed", "manual_batch_02"),
    # draft rules for the remaining comments, to be confirmed
    CommentRule(r"\bnot a live performance\b", "mismatch",
                [_rec("other", "live performance", "studio-like", "incorrect")], source="manual_draft_03"),
    CommentRule(r"confounds afrobeat", "mismatch",
                [_rec("genre", "afrobeat", "generic African-themed music", "incorrect")], source="manual_draft_03"),
    CommentRule(r"very far stylistically from the prompted", "mismatch",
                [_rec("genre", "prompted style", "different style", "incorrect")], source="manual_draft_03"),
    CommentRule(r"\bnot classical hindustani\b", "mismatch",
                [_rec("genre", "Hindustani classical", "non-classical style", "incorrect")], source="manual_draft_03"),
    CommentRule(r"\bno (?:pormted|prompted) noises?\b", "mismatch",
                [_rec("other", "prompted noises / sound effects", "absent", "missing")], source="manual_draft_03"),
    CommentRule(r"(?:^|[.!]\s*)no vocals?\b", "mismatch",
                [_rec("vocals", "vocals", "absent", "missing")], source="manual_draft_03"),
    CommentRule(r"\bnot clearly beatbox\b", "mismatch",
                [_rec("vocals", "beatbox", "unclear beatbox", "incorrect")], source="manual_draft_03"),
    CommentRule(r"instruments present although sparse", "mismatch",
                [_rec("instruments", "no instruments (a cappella)", "sparse instruments present", "added")],
                source="manual_draft_03", notes="Check the full comment: 'ignoring ...' which instruction?"),
    CommentRule(r"\bsome words added\b", "mismatch",
                [_rec("vocals", "no words (vocables only)", "some words added", "added")], source="manual_draft_03"),
    CommentRule(r"\bno people talking or cla(?:p)?ping\b", "mismatch",
                [_rec("other", "people talking and clapping", "absent", "missing")], source="manual_draft_03"),
    CommentRule(r"does not follow [\"']?no singer", "mismatch",
                [_rec("vocals", "no singer", "vocals present", "added")], source="manual_draft_03"),
    CommentRule(r"\bnot an? am(?:a)?teur recording\b", "mismatch",
                [_rec("other", "amateur recording", "professional-sounding", "incorrect")], source="manual_draft_03"),
    CommentRule(r"\bnot a classical melody\b", "mismatch",
                [_rec("genre", "classical melody", "non-classical", "incorrect")], source="manual_draft_03"),
    CommentRule(r"follows (?:well )?the low quality and weird prompts", "positive", [], source="manual_draft_03"),
    CommentRule(r"follows well the dutch language prompt", "positive", [], source="manual_draft_03"),
    CommentRule(r"best version that follows the prompt", "positive", [], source="manual_draft_03"),
    CommentRule(r"catchy known riff", "descriptive", [], source="manual_draft_03",
                notes="Possible memorisation; not a prompt mismatch."),
    # manual review 04: gaps found reading the full comments
    CommentRule(r"does not follow the vocal command and makes it instrumental", "mismatch",
                [_rec("vocals", "vocals", "absent", "missing")], "text_reviewed", "manual_review_04"),
    CommentRule(r"instead using a guitar", "mismatch",
                [_rec("instruments", "piano and synth", "guitar", "replaced")], "text_reviewed", "manual_review_04"),
    CommentRule(r"nothing polish in it|stilistically innacurrate", "mismatch",
                [_rec("genre", "polish folk music", "irish folk / stylistically inaccurate", "incorrect")],
                "text_reviewed", "manual_review_04"),
    CommentRule(r"\bno humming\b", "mismatch",
                [_rec("vocals", "humming", "absent", "missing")], "text_reviewed", "manual_review_04"),
    # wider than rule_v2 voice rules; their matches are skipped
    CommentRule(r"\bnot clearly (?:a )?(?P<gender>male|female) (?:voice|vocal|singer)s?\b", "mismatch",
                [_rec("vocals", "{gender} vocals", "unclear vocal type", "incorrect")],
                "text_reviewed", "manual_review_04"),
    CommentRule(r"lyrics generated in \w+ in response to [\"']?foreign language", "descriptive", [],
                "text_reviewed", "manual_review_04",
                "The prompt only asks for a foreign language, so Spanish / French lyrics satisfy it."),
    CommentRule(r"could slightly pass as low quality", "mismatch",
                [_rec("other", "low recording quality", "far from low quality", "incorrect")],
                "text_reviewed", "manual_review_04", "'still far away from low'"),
    CommentRule(r"lab[ae]l+ed instrumental by the generator but has", "mismatch",
                [_rec("vocals", "vocals not specified", "lyrics added", "unexpected_addition")],
                "text_reviewed", "manual_review_04", "The prompt has no vocals / instrumental requirement."),
    CommentRule(r"foreign hindi accent", "mismatch",
                [_rec("vocals", "accent not specified", "Hindi accent added", "unexpected_addition")],
                "text_reviewed", "manual_review_04"),
    CommentRule(r"composition are not followed", "mismatch",
                [_rec("other", "composition as described (what happens in which bar)", "not followed", "incorrect")],
                "text_reviewed", "manual_review_04"),
]

# generic fallback: "no <instrument>" for instruments not covered above
_GENERIC_NO_INSTRUMENT = re.compile(r"(?:^|[.,;]\s*)no ([a-z][a-z \-]{2,30}?)(?=[.,;]|$)")


def _generic_instrument_records(text: str, covered_spans: list[tuple[int, int]]):
    from .vocab import INSTRUMENT_TERMS  # local import to avoid a cycle

    out = []
    for m in _GENERIC_NO_INSTRUMENT.finditer(text):
        if any(s <= m.start(1) < e for s, e in covered_spans):
            continue
        phrase = m.group(1).strip()
        if any(re.fullmatch(rf"(?:\w+ )?{t}s?", phrase) for t in INSTRUMENT_TERMS):
            out.append(dict(_rec("instruments", phrase, "absent", "missing"),
                            evidence=m.group(0).strip(" .,;"), span=(m.start(1), m.end(1)),
                            source="generic_no_instrument", review_status="needs_review"))
    return out


def rule_v2_records(text: str) -> list[dict]:
    """records from the notebook-02 rules alone, before review 04."""
    records, seen = [], set()
    for rule in RULES_V2:
        for m in re.finditer(rule["pattern"], text):
            key = (rule["category"], rule["expected"], rule["error_type"], m.start(), m.end())
            if key in seen:
                continue
            seen.add(key)
            records.append(dict(category=rule["category"], expected=rule["expected"],
                                observed=rule["observed"], error_type=rule["error_type"],
                                evidence=m.group(0), source="rule_v2", review_status="auto_extracted",
                                pattern=rule["pattern"], span=(m.start(), m.end())))
    return records


def extract_records(comment) -> tuple[list[dict], str]:
    """return (records, comment_type) for one expert comment."""
    if comment is None or (isinstance(comment, float) and pd.isna(comment)) or not str(comment).strip():
        return [], "missing"
    text = str(comment).lower().strip()
    records, spans = [], []
    types = []

    for rec in rule_v2_records(text):
        spans.append(rec.pop("span"))
        fix = RULE_V2_FIXES.get(rec.pop("pattern"))
        if fix:
            rec.update(fix, source="manual_review_04", review_status="text_reviewed",
                       review_notes="extracted by rule_v2; error_type changed from incorrect to added")
        records.append(rec)
        types.append("mismatch")

    for cr in COMMENT_RULES:
        m = re.search(cr.pattern, text)
        if not m:
            continue
        # skip a comment rule fully inside an already-extracted span
        if any(s <= m.start() and m.end() <= e for s, e in spans):
            continue
        spans.append((m.start(), m.end()))
        types.append(cr.comment_type)
        groups = m.groupdict()
        for r in cr.records:
            rec = {k: (v.format(**groups) if isinstance(v, str) and groups else v) for k, v in r.items()}
            key = (rec["category"], rec["expected"], rec["error_type"])
            if any((x["category"], x["expected"], x["error_type"]) == key for x in records):
                continue
            records.append(dict(rec, evidence=m.group(0), source=cr.source,
                                review_status=cr.review_status, review_notes=cr.notes))

    for rec in _generic_instrument_records(text, spans):
        if not any(r["category"] == "instruments" and rec["expected"] in str(r["expected"]) for r in records):
            rec.pop("span", None)
            records.append(rec)
            types.append("mismatch")

    if not types:
        comment_type = "unparsed"
    elif all(t == "positive" for t in types):
        comment_type = "positive"
    elif "positive" in types or "mixed" in types:
        comment_type = "mixed" if any(t in ("mismatch", "mixed") for t in types) else "positive"
    elif all(t == "descriptive" for t in types):
        comment_type = "descriptive"
    else:
        comment_type = "mismatch"
    if not records and comment_type == "mixed":
        comment_type = "positive"
    return records, comment_type


def build_mismatch_records(df: pd.DataFrame) -> pd.DataFrame:
    """one row per (track, extracted issue)."""
    rows = []
    for _, row in df.iterrows():
        records, _ = extract_records(row.get("Comments"))
        for rec in records:
            rec = dict(rec)
            if (row["original_id"] in UNEXPECTED_VOCALS_ORIGINAL_IDS and rec["expected"] == "no vocals"
                    and "added voice" in rec["evidence"]):
                rec.update(expected="vocals not specified", observed="vocals added",
                           error_type="unexpected_addition", review_status="needs_audio_review",
                           review_notes="Vocals were not explicitly prohibited; audio verification required.")
            rows.append(dict(track_id=row["track_id"], original_id=row["original_id"],
                             prompt_type=row["prompt_type"], expert_comment=row["Comments"], **rec))
    cols = ["track_id", "original_id", "prompt_type", "expert_comment", "category", "expected",
            "observed", "error_type", "evidence", "source", "review_status", "review_notes"]
    out = pd.DataFrame(rows, columns=cols)
    out["verdict"] = out["error_type"].map(ERROR_TYPE_TO_VERDICT)
    out.insert(0, "annotation_id", range(1, len(out) + 1))
    return out


ERROR_TYPE_TO_VERDICT = {
    "missing": "ignored",
    "added": "ignored",
    "incorrect": "changed",
    "replaced": "changed",
    "unexpected_addition": "hallucinated",
    "partial": "partially_met",
}

MISMATCH_ERROR_TYPES = {"missing", "added", "incorrect", "replaced", "unexpected_addition"}

VOCALS_PRESENT_PATTERNS = [r"\badded voice\b", r"\bnot instrumental\b", r"does not follow [\"']?instrumental",
                           r"lab[ae]l+ed instrumental by the generator but has", r"does not follow [\"']?no singer",
                           r"\bsome words added\b", r"lyrics generated in", r"realistic voice", r"\bonly vocables\b"]
VOCALS_ABSENT_PATTERNS = [r"(?:^|[.!]\s*)no vocals?\b", r"\bdoes not have (?:a )?voice\b"]


def vocals_from_comment(comment) -> float | None:
    """vocal presence stated in the comment: 1.0, 0.0 or None."""
    if comment is None or (isinstance(comment, float) and pd.isna(comment)):
        return None
    text = str(comment).lower()
    if any(re.search(p, text) for p in VOCALS_ABSENT_PATTERNS):
        return 0.0
    if any(re.search(p, text) for p in VOCALS_PRESENT_PATTERNS):
        return 1.0
    return None


def build_ground_truth(df: pd.DataFrame, records: pd.DataFrame | None = None,
                       count_partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """return ``(per_category, per_track)`` ground-truth tables."""
    if records is None:
        records = build_mismatch_records(df)
    error_types = set(MISMATCH_ERROR_TYPES) | ({"partial"} if count_partial else set())
    rec = records[records["error_type"].isin(error_types)]

    commented = has_text(df["Comments"])
    comment_type = df["Comments"].apply(lambda c: extract_records(c)[1])

    rows = []
    for (_, row), has_c in zip(df.iterrows(), commented):
        for cat in CATEGORIES:
            r = rec[(rec["track_id"] == row["track_id"]) & (rec["category"] == cat)]
            if len(r):
                src, y = "expert_record", 1
            elif has_c:
                src, y = "comment_no_issue", 0
            else:
                src, y = "no_comment_assumed_ok", 0
            rows.append(dict(
                track_id=row["track_id"], original_id=row["original_id"], prompt_type=row["prompt_type"],
                category=cat, gt_mismatch=y, gt_source=src,
                gt_verdicts=";".join(sorted(set(r["verdict"].dropna()))),
                gt_expected=" | ".join(r["expected"].astype(str)),
                gt_needs_review=bool(r["review_status"].isin(["needs_review", "needs_audio_review"]).any()),
            ))
    per_category = pd.DataFrame(rows)

    per_track = df[["track_id", "original_id", "generation_version", "prompt_type"]].copy()
    per_track["has_comment"] = commented.values
    per_track["comment_type"] = comment_type.values
    any_mm = per_category.groupby("track_id")["gt_mismatch"].max()
    per_track["any_mismatch"] = per_track["track_id"].map(any_mm).astype(int)
    per_track["n_records"] = per_track["track_id"].map(rec.groupby("track_id").size()).fillna(0).astype(int)

    from_comment = df["Comments"].apply(vocals_from_comment)
    if "Instrumental" in df:
        flag = df["Instrumental"].astype(str).str.lower().map({"true": 0.0, "false": 1.0})
    else:
        flag = pd.Series(float("nan"), index=df.index)
    per_track["vocals_present_gt"] = from_comment.where(from_comment.notna(), flag).values
    per_track["vocals_gt_source"] = (
        pd.Series("comment", index=df.index).where(from_comment.notna(), "instrumental_flag").values
    )
    if "emotion_clean" in df:
        per_track["emotion_gt"] = df["emotion_clean"].values
    if "extra_emotion_list" in df:
        per_track["extra_emotions_gt"] = df["extra_emotion_list"].apply(";".join).values
    return per_category, per_track


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """track-level coverage of commented tracks, by comment_type."""
    commented = df[has_text(df["Comments"])]
    ct = commented["Comments"].apply(lambda c: extract_records(c)[1])
    return ct.value_counts().rename_axis("comment_type").to_frame("tracks")
