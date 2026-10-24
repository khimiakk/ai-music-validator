"""Stage 4: requirements + cached CLAP audio embeddings -> one verdict per requirement.

Verdicts: satisfied | ignored | changed | hallucinated (see README). Helper verdicts that never
count as a mismatch: not_checked (gender when no vocals are heard), unknown (unmapped tempo word)
and none (the hallucination check found no vocals).

Checks
------
* tempo           CLAP zero-shot slow / medium / fast + prior correction (tempo_word_verdict);
                  an explicit BPM would use tempo_verdict on the librosa estimate
* vocal_presence  p_vocals = CLAP softmax("a song with singing vocals" vs "instrumental music
                  without vocals"); +1 and p < 0.5 -> ignored, -1 and p >= 0.5 -> ignored
* vocal_gender    only if p_vocals >= 0.5; female vs male vocalist, wrong gender -> changed
* everything else (check "clap" and "essentia_*"; Essentia is postponed): s = cos(track, text),
                  score_pct = percentile of s among tracks with no requirement of the same
                  category and value; satisfied if score_pct >= tau, reversed for polarity -1
* hallucination   p_vocals >= 0.5 in a track whose prompt has no vocals requirement at all

Every row gets a mismatch_score in [0, 1] (higher = more likely a mismatch) for ROC-AUC:
1 - score_pct for CLAP checks (score_pct for prohibitions), the CLAP probability of the wrong
answer for the vocal checks, and 0/1 for the rule-based tempo check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import CATEGORIES
from .audio.clap import track_embedding
from .audio.tempo import TEMPO_CLASSES, tempo_verdict, tempo_word_verdict
from .data import PROMPT_TYPE_BY_VERSION

VOCAL_TEXTS = ["a song with singing vocals", "instrumental music without vocals"]
GENDER_TEXTS = ["a song sung by a female vocalist", "a song sung by a male vocalist"]
TEMPO_TEXTS = [f"{c} tempo music" for c in TEMPO_CLASSES]
TEMPO_COLS = [f"p_{c}" for c in TEMPO_CLASSES]
RULE_CHECKS = {"tempo", "vocal_presence", "vocal_gender"}  # every other check is a calibrated CLAP check
MISMATCH_VERDICTS = {"ignored", "changed", "hallucinated"}
DEFAULT_TAU = 0.5


def stack_track_embeddings(window_embs) -> tuple[list[str], np.ndarray]:
    """Track embedding = normalised mean of the window embeddings. Returns (track_ids, matrix)."""
    ids = sorted(window_embs)
    return ids, np.stack([track_embedding(window_embs[t]) for t in ids])


def softmax_probs(audio: np.ndarray, texts: np.ndarray, logit_scale: float) -> np.ndarray:
    """(n_tracks, n_texts): softmax over the texts of logit_scale * cosine similarity."""
    logits = logit_scale * audio @ texts.T
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def audio_evidence(track_ids, E, embed_text, logit_scale) -> pd.DataFrame:
    """Per track: p_vocals, p_female and the slow / medium / fast tempo probabilities."""
    vocals = softmax_probs(E, embed_text(VOCAL_TEXTS), logit_scale)
    gender = softmax_probs(E, embed_text(GENDER_TEXTS), logit_scale)
    tempo = softmax_probs(E, embed_text(TEMPO_TEXTS), logit_scale)
    ev = pd.DataFrame({"p_vocals": vocals[:, 0], "p_female": gender[:, 0]},
                      index=pd.Index(track_ids, name="track_id"))
    ev[TEMPO_COLS] = tempo
    return ev


def presence_text(clap_text: str, value: str, polarity: int) -> str:
    """CLAP scores the element itself, also for prohibitions (it handles "without ..." badly)."""
    return clap_text if polarity > 0 else f"music with {value}"


def clap_percentiles(reqs: pd.DataFrame, track_ids, E, embed_text) -> pd.DataFrame:
    """s and score_pct for every CLAP-check requirement (index = reqs index)."""
    clap = reqs[~reqs["check"].isin(RULE_CHECKS)].copy()
    clap["text"] = [presence_text(t, v, p) for t, v, p in zip(clap["clap_text"], clap["value"], clap["polarity"])]
    texts = sorted(set(clap["text"]))
    text_emb = dict(zip(texts, embed_text(texts)))
    pos = {t: i for i, t in enumerate(track_ids)}
    mentioned = reqs.groupby(["category", "value"])["track_id"].agg(set)

    clap["s"], clap["score_pct"] = np.nan, np.nan
    for (text, cat, value), g in clap.groupby(["text", "category", "value"]):
        sims = E @ text_emb[text]
        ref = sims[[pos[t] for t in track_ids if t not in mentioned[(cat, value)]]]
        own = sims[[pos[t] for t in g["track_id"]]]
        clap.loc[g.index, "s"] = own
        clap.loc[g.index, "score_pct"] = [(ref < s).mean() + 0.5 * (ref == s).mean() for s in own]
    return clap[["text", "s", "score_pct"]]


def clap_verdict(score_pct: float, polarity: int, tau: float = DEFAULT_TAU) -> str:
    """satisfied if the element is detected (score_pct >= tau) and wanted, or absent and prohibited."""
    present = score_pct >= tau
    return "satisfied" if present == (polarity > 0) else "ignored"


def requirement_verdicts(reqs: pd.DataFrame, evidence: pd.DataFrame, clap_scores: pd.DataFrame,
                         class_means, tau: float = DEFAULT_TAU, bpm: dict | None = None) -> pd.DataFrame:
    """One row per requirement with verdict, detail, prob / score_pct and mismatch_score."""
    rows = []
    for idx, r in reqs.iterrows():
        ev = evidence.loc[r["track_id"]]
        out = dict(verdict="unknown", detail="", prob=np.nan, s=np.nan, score_pct=np.nan, mismatch_score=0.0)
        if r["check"] == "tempo":
            if r["value"].endswith("bpm") and bpm is not None:
                verdict, detail = tempo_verdict(bpm.get(r["track_id"], np.nan), r["params"]["bpm_min"],
                                                r["params"]["bpm_max"])
            else:
                verdict, detail = tempo_word_verdict(ev[TEMPO_COLS].to_numpy(float), r["value"], class_means)
            out.update(verdict=verdict, detail=detail, mismatch_score=float(verdict == "changed"))
        elif r["check"] == "vocal_presence":
            p = ev["p_vocals"]
            heard = p >= 0.5
            ok = heard if r["polarity"] > 0 else not heard
            out.update(verdict="satisfied" if ok else "ignored", prob=p,
                       mismatch_score=1 - p if r["polarity"] > 0 else p)
        elif r["check"] == "vocal_gender":
            if ev["p_vocals"] < 0.5:
                out.update(verdict="not_checked", detail="no vocals heard")
            else:
                p_expected = ev["p_female"] if r["params"]["gender"] == "female" else 1 - ev["p_female"]
                out.update(verdict="satisfied" if p_expected >= 0.5 else "changed", prob=p_expected,
                           mismatch_score=1 - p_expected)
        else:
            pct = clap_scores.loc[idx, "score_pct"]
            out.update(verdict=clap_verdict(pct, r["polarity"], tau), s=clap_scores.loc[idx, "s"],
                       score_pct=pct, mismatch_score=1 - pct if r["polarity"] > 0 else pct)
        rows.append(out)
    return pd.concat([reqs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def hallucination_rows(reqs: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """One row per track whose prompt has no vocals requirement: hallucinated if vocals are heard."""
    with_vocals = set(reqs.loc[reqs["category"] == "vocals", "track_id"])
    rows = []
    for tid, p in evidence["p_vocals"].items():
        if tid in with_vocals:
            continue
        rows.append(dict(req_id=f"{tid}#vocals_not_in_prompt", track_id=tid, original_id=tid.rsplit("_", 1)[0],
                         prompt_type=PROMPT_TYPE_BY_VERSION[tid.rsplit("_", 1)[1]], category="vocals",
                         value="vocals (not in prompt)", polarity=0, check="vocal_hallucination",
                         verdict="hallucinated" if p >= 0.5 else "none", prob=p, mismatch_score=p))
    return pd.DataFrame(rows)


def all_verdicts(reqs, evidence, clap_scores, class_means, tau=DEFAULT_TAU, bpm=None) -> pd.DataFrame:
    return pd.concat([requirement_verdicts(reqs, evidence, clap_scores, class_means, tau, bpm),
                      hallucination_rows(reqs, evidence)], ignore_index=True)


def apply_tau(verdicts: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Same verdicts with a different tau for the CLAP checks (rule-based checks are unchanged)."""
    v = verdicts.copy()
    clap = v["score_pct"].notna()
    present = v.loc[clap, "score_pct"] >= tau
    v.loc[clap, "verdict"] = np.where(present == (v.loc[clap, "polarity"] > 0), "satisfied", "ignored")
    return v


def predictions_per_category(verdicts: pd.DataFrame, track_ids) -> pd.DataFrame:
    """One row per (track, category): pred_mismatch = any ignored / changed / hallucinated verdict,
    pred_score = max mismatch_score (both 0 when the prompt has nothing in that category)."""
    v = verdicts.assign(is_mismatch=verdicts["verdict"].isin(MISMATCH_VERDICTS),
                        is_requirement=verdicts["check"] != "vocal_hallucination")
    agg = v.groupby(["track_id", "category"]).agg(n_requirements=("is_requirement", "sum"),
                                                  pred_mismatch=("is_mismatch", "any"),
                                                  pred_score=("mismatch_score", "max"))
    grid = pd.MultiIndex.from_product([list(track_ids), CATEGORIES], names=["track_id", "category"])
    out = agg.reindex(grid).reset_index()
    out["n_requirements"] = out["n_requirements"].fillna(0).astype(int)
    out["pred_mismatch"] = out["pred_mismatch"].fillna(False).astype(bool)
    out["pred_score"] = out["pred_score"].fillna(0.0)
    return out
