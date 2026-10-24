"""Metrics with 95% confidence intervals from a cluster bootstrap over original_id.

The 4 versions of a prompt share the prompt and usually the expert comment, so they are not
independent: every bootstrap resample draws whole original_ids (with replacement).
Thresholds that are tuned (decision threshold, tau) are chosen with GroupKFold over original_id
and evaluated on the held-out fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

N_BOOT = 1000
MIN_POSITIVES = 5  # fewer positives in a category -> "not enough labels"
METRICS = ["precision", "recall", "f1", "roc_auc"]


def binary_metrics(y, pred, score=None) -> dict:
    """precision / recall / F1 of the binary predictions, ROC-AUC of the score, support = positives."""
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()
    auc = np.nan
    if score is not None and 0 < y.sum() < len(y):
        auc = roc_auc_score(y, score)
    return dict(precision=tp / (tp + fp) if tp + fp else np.nan, recall=tp / (tp + fn) if tp + fn else np.nan,
                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan, roc_auc=auc,
                support=int(y.sum()), n=len(y))


def cluster_bootstrap(stat_fn, groups, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Call stat_fn(row_indices) on n_boot resamples of whole groups; one row of statistics each."""
    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        pick = rng.integers(len(members), size=len(members))
        rows.append(stat_fn(np.concatenate([members[i] for i in pick])))
    return pd.DataFrame(rows)


def with_ci(point: dict, boot: pd.DataFrame, keys) -> dict:
    """{key: value, key_lo: 2.5%, key_hi: 97.5%} for the given keys."""
    out = {}
    for k in keys:
        out[k] = point[k]
        vals = boot[k].to_numpy(float) if k in boot else np.array([np.nan])
        ok = vals[~np.isnan(vals)]
        out[f"{k}_lo"], out[f"{k}_hi"] = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    return out


def metrics_with_ci(y, pred, score, groups, n_boot=N_BOOT, seed=0) -> dict:
    """binary_metrics + 95% cluster-bootstrap CIs."""
    y, pred, score = np.asarray(y), np.asarray(pred), np.asarray(score, float)
    boot = cluster_bootstrap(lambda i: binary_metrics(y[i], pred[i], score[i]), groups, n_boot, seed)
    point = binary_metrics(y, pred, score)
    return dict(with_ci(point, boot, METRICS), support=point["support"], n=point["n"])


def category_metrics(cells: pd.DataFrame, categories, n_boot=N_BOOT, seed=0) -> pd.DataFrame:
    """Per category + micro + macro (over categories with >= MIN_POSITIVES positives).

    cells: one row per (track, category) with gt_mismatch, pred_mismatch, pred_score, original_id.
    """
    y = cells["gt_mismatch"].to_numpy(bool)
    pred = cells["pred_mismatch"].to_numpy(bool)
    score = cells["pred_score"].to_numpy(float)
    cat = cells["category"].to_numpy()
    enough = [c for c in categories if y[cat == c].sum() >= MIN_POSITIVES]

    def stats(idx):
        out = {}
        for c in categories:
            m = idx[cat[idx] == c]
            out.update({(c, k): v for k, v in binary_metrics(y[m], pred[m], score[m]).items()})
        m = idx[np.isin(cat[idx], enough)]
        out.update({("micro", k): v for k, v in binary_metrics(y[m], pred[m], score[m]).items()})
        for k in METRICS:
            out[("macro", k)] = np.nanmean([out[(c, k)] for c in enough]) if enough else np.nan
        return out

    point = stats(np.arange(len(y)))
    boot = cluster_bootstrap(stats, cells["original_id"], n_boot, seed)
    rows = []
    for c in list(categories) + ["micro", "macro"]:
        p = {k: point[(c, k)] for k in METRICS}
        b = boot[[(c, k) for k in METRICS]].set_axis(METRICS, axis=1)
        if c == "macro":
            support, n = sum(point[(e, "support")] for e in enough), sum(point[(e, "n")] for e in enough)
        else:
            support, n = point[(c, "support")], point[(c, "n")]
        note = "" if c in enough or c in ("micro", "macro") else "not enough labels"
        rows.append(dict(category=c, support=support, n=n, **with_ci(p, b, METRICS), note=note))
    return pd.DataFrame(rows)


def best_f1_threshold(y, score) -> float:
    """Threshold (predict score >= t) with the highest F1 on these rows."""
    y = np.asarray(y, bool)
    candidates = np.unique(score)
    f1s = [binary_metrics(y, score >= t)["f1"] for t in candidates]
    return float(candidates[int(np.nanargmax(f1s))])


def oof_threshold_predictions(y, score, groups, n_splits: int = 5):
    """Out-of-fold predictions: the threshold is chosen on 4 folds (GroupKFold) and applied to the 5th."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    pred, thresholds = np.zeros(len(y), bool), []
    for train, test in GroupKFold(n_splits=n_splits).split(np.zeros((len(y), 1)), y, groups):
        t = best_f1_threshold(y[train], score[train])
        thresholds.append(t)
        pred[test] = score[test] >= t
    return pred, thresholds


def multiclass_metrics_with_ci(y_true, y_pred, labels, groups, n_boot=N_BOOT, seed=0) -> dict:
    """Accuracy and macro-F1 (over `labels`) with 95% cluster-bootstrap CIs."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def stats(i):
        return dict(accuracy=float(np.mean(y_true[i] == y_pred[i])),
                    macro_f1=f1_score(y_true[i], y_pred[i], labels=labels, average="macro", zero_division=0))

    point = stats(np.arange(len(y_true)))
    boot = cluster_bootstrap(stats, groups, n_boot, seed)
    return dict(with_ci(point, boot, ["accuracy", "macro_f1"]), n=len(y_true))


def fmt_ci(row, key, digits=2) -> str:
    """'0.62 [0.51, 0.71]' (or '–' when undefined)."""
    v = row[key]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    lo, hi = row.get(f"{key}_lo", np.nan), row.get(f"{key}_hi", np.nan)
    if np.isnan(lo):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
