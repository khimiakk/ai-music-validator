"""stage 4: one verdict per requirement from cached audio evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import CATEGORIES
from .audio.clap import track_embedding
from .audio.tempo import TEMPO_CLASSES, tempo_verdict, tempo_word_verdict
from .data import PROCESSED_DIR, PROMPT_TYPE_BY_VERSION

VOCAL_TEXTS = ["a song with singing vocals", "instrumental music without vocals"]
GENDER_TEXTS = ["a song sung by a female vocalist", "a song sung by a male vocalist"]
TEMPO_TEXTS = [f"{c} tempo music" for c in TEMPO_CLASSES]
TEMPO_COLS = [f"p_{c}" for c in TEMPO_CLASSES]
RULE_CHECKS = {"tempo", "vocal_presence", "vocal_gender"}  # every other check is a calibrated CLAP check
MISMATCH_VERDICTS = {"ignored", "changed", "hallucinated"}
DEFAULT_TAU = 0.5

ESSENTIA_CSV = PROCESSED_DIR / "essentia_scores.csv"
ESSENTIA_PREFIX = {"essentia_genre": "genre", "essentia_instrument": "instr", "essentia_mood": "mood"}
SYSTEMS = ["clap", "essentia", "average"]


def stack_track_embeddings(window_embs) -> tuple[list[str], np.ndarray]:
    """track embedding = normalised mean of its window embeddings."""
    ids = sorted(window_embs)
    return ids, np.stack([track_embedding(window_embs[t]) for t in ids])


def softmax_probs(audio: np.ndarray, texts: np.ndarray, logit_scale: float) -> np.ndarray:
    """softmax over texts of logit_scale * cosine similarity."""
    logits = logit_scale * audio @ texts.T
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def audio_evidence(track_ids, E, embed_text, logit_scale) -> pd.DataFrame:
    """per track: p_vocals, p_female and the tempo class probabilities."""
    vocals = softmax_probs(E, embed_text(VOCAL_TEXTS), logit_scale)
    gender = softmax_probs(E, embed_text(GENDER_TEXTS), logit_scale)
    tempo = softmax_probs(E, embed_text(TEMPO_TEXTS), logit_scale)
    ev = pd.DataFrame({"p_vocals": vocals[:, 0], "p_female": gender[:, 0]},
                      index=pd.Index(track_ids, name="track_id"))
    ev[TEMPO_COLS] = tempo
    return ev


def presence_text(clap_text: str, value: str, polarity: int) -> str:
    """text to score: the element itself, also for prohibitions."""
    return clap_text if polarity > 0 else f"music with {value}"


def percentile(s: float, ref: np.ndarray) -> float:
    """share of reference scores below s, ties count half."""
    if np.isnan(s):
        return np.nan
    return float((ref < s).mean() + 0.5 * (ref == s).mean())


def clap_percentiles(reqs: pd.DataFrame, track_ids, E, embed_text) -> pd.DataFrame:
    """s and score_pct for every CLAP-check requirement."""
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
        clap.loc[g.index, "score_pct"] = [percentile(s, ref) for s in own]
    return clap[["text", "s", "score_pct"]]


def load_essentia_scores(path=ESSENTIA_CSV, stat: str = "mean") -> pd.DataFrame | None:
    """per-track Essentia activations for one statistic, or None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path).set_index("track_id")
    cols = [c for c in df.columns if c.endswith(f"__{stat}")]
    return df[cols].rename(columns=lambda c: c[: -len(stat) - 2])


def essentia_track_scores(essentia: pd.DataFrame, check: str, params: dict) -> pd.Series | None:
    """one Essentia score per track, or None if label unknown."""
    prefix = ESSENTIA_PREFIX[check]
    if check == "essentia_genre":
        cols = [c for c in essentia.columns if c.startswith(f"{prefix}__") and any(s in c for s in params["labels"])]
        return essentia[cols].max(axis=1) if cols else None
    col = f"{prefix}__{params['label']}"
    return essentia[col] if col in essentia.columns else None


def essentia_percentiles(reqs: pd.DataFrame, essentia: pd.DataFrame, track_ids) -> pd.DataFrame:
    """s and score_pct for essentia_* requirements, calibrated like CLAP."""
    rows = reqs[reqs["check"].isin(ESSENTIA_PREFIX)]
    mentioned = reqs.groupby(["category", "value"])["track_id"].agg(set)
    out = pd.DataFrame(np.nan, index=rows.index, columns=["s", "score_pct"])
    for (check, cat, value), g in rows.groupby(["check", "category", "value"]):
        scores = essentia_track_scores(essentia, check, g["params"].iloc[0])
        if scores is None:
            continue
        ref = scores.reindex([t for t in track_ids if t not in mentioned[(cat, value)]]).dropna().to_numpy()
        own = scores.reindex(g["track_id"]).to_numpy(float)
        out.loc[g.index, "s"] = own
        out.loc[g.index, "score_pct"] = [percentile(s, ref) for s in own]
    return out


def add_essentia_voice(evidence: pd.DataFrame, essentia: pd.DataFrame) -> pd.DataFrame:
    """add p_voice, the Essentia probability of "voice", to evidence."""
    ev = evidence.copy()
    ev["p_voice"] = essentia["voice__voice"].reindex(ev.index) if "voice__voice" in essentia.columns else np.nan
    return ev


def system_inputs(system: str, evidence: pd.DataFrame, clap_scores: pd.DataFrame,
                  essentia_scores: pd.DataFrame | None = None):
    """evidence and scores for the clap, essentia or average system."""
    ev, scores = evidence.copy(), clap_scores[["s", "score_pct"]].copy()
    if system == "clap" or essentia_scores is None:
        return ev, scores
    ess = essentia_scores.reindex(scores.index)
    has = ess["score_pct"].notna()
    p_voice = ev["p_voice"] if "p_voice" in ev.columns else pd.Series(np.nan, index=ev.index)
    if system == "essentia":
        scores.loc[has, ["s", "score_pct"]] = ess.loc[has, ["s", "score_pct"]]
        ev["p_vocals"] = p_voice.fillna(ev["p_vocals"])
    elif system == "average":
        scores.loc[has, "score_pct"] = (scores.loc[has, "score_pct"] + ess.loc[has, "score_pct"]) / 2
        scores.loc[has, "s"] = np.nan  # not comparable across the two models
        ev["p_vocals"] = ((ev["p_vocals"] + p_voice) / 2).fillna(ev["p_vocals"])
    else:
        raise ValueError(f"unknown system {system!r}, expected one of {SYSTEMS}")
    return ev, scores


def clap_verdict(score_pct: float, polarity: int, tau: float = DEFAULT_TAU) -> str:
    """satisfied if detection (score_pct >= tau) matches the polarity."""
    present = score_pct >= tau
    return "satisfied" if present == (polarity > 0) else "ignored"


def requirement_verdicts(reqs: pd.DataFrame, evidence: pd.DataFrame, clap_scores: pd.DataFrame,
                         class_means, tau: float = DEFAULT_TAU, bpm: dict | None = None) -> pd.DataFrame:
    """one row per requirement with its verdict and mismatch_score."""
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
    """hallucination check for tracks without any vocals requirement."""
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
    """same verdicts with a different tau for the CLAP checks."""
    v = verdicts.copy()
    clap = v["score_pct"].notna()
    present = v.loc[clap, "score_pct"] >= tau
    v.loc[clap, "verdict"] = np.where(present == (v.loc[clap, "polarity"] > 0), "satisfied", "ignored")
    return v


def predictions_per_category(verdicts: pd.DataFrame, track_ids) -> pd.DataFrame:
    """one row per (track, category) with pred_mismatch and pred_score."""
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
