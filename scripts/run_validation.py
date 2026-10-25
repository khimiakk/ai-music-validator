"""Stage 4: verdict per requirement, CLAP baseline, evaluation against the expert labels.

    run_validation.py                 # everything (CLAP is loaded for TEXT embeddings only)
    run_validation.py --n-boot 200    # quicker bootstrap while trying things out

Inputs (all cached, no audio): data/processed/clap_audio_windows.npz, clap_meta.json,
audio_features.csv, requirements.csv, ground_truth_per_category.csv, ground_truth_per_track.csv,
mismatch_records.csv (+ the raw CSV for the full prompt text).

Outputs: data/processed/requirement_verdicts.csv, predictions_per_category.csv,
results/metrics_*.csv, results/figures/*.png, results/summary.md.
All CIs are 95% cluster-bootstrap intervals over original_id (see aimv.evaluation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimv import CATEGORIES  # noqa: E402
from aimv.audio.clap import ClapScorer  # noqa: E402
from aimv.audio.tempo import TEMPO_CLASSES, TEMPO_WORD_CLASSES, prior_correct  # noqa: E402
from aimv.data import PROCESSED_DIR, PROJECT_ROOT, load_sunocaps  # noqa: E402
from aimv.evaluation import (MIN_POSITIVES, binary_metrics, category_metrics, cluster_bootstrap,  # noqa: E402
                             fmt_ci, metrics_with_ci, multiclass_metrics_with_ci, oof_threshold_predictions,
                             paired_difference, with_ci)
from aimv.verdicts import (DEFAULT_TAU, ESSENTIA_CSV, ESSENTIA_PREFIX, SYSTEMS, TEMPO_COLS,  # noqa: E402
                           add_essentia_voice, all_verdicts, apply_tau, audio_evidence, clap_percentiles,
                           essentia_percentiles, load_essentia_scores, predictions_per_category, softmax_probs,
                           stack_track_embeddings, system_inputs)
from aimv.vocab import SUNOCAPS_EMOTIONS  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
TAUS = np.round(np.arange(0.30, 0.801, 0.05), 2)
SHOWN_VERDICTS = ["satisfied", "ignored", "changed", "hallucinated"]
GT_COLS = ["track_id", "category", "original_id", "prompt_type", "gt_mismatch", "gt_source", "gt_needs_review"]
# chart colours (light theme): surface, text, muted text, grid, categorical slots 1-5
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# ---------------------------------------------------------------------------
# inputs and verdicts
# ---------------------------------------------------------------------------

def load_inputs():
    reqs = pd.read_csv(PROCESSED_DIR / "requirements.csv")
    reqs["params"] = reqs["params"].map(json.loads)
    df = load_sunocaps().set_index("track_id")
    return dict(
        meta=json.load(open(PROCESSED_DIR / "clap_meta.json")),
        window_embs=dict(np.load(PROCESSED_DIR / "clap_audio_windows.npz")),
        bpm=pd.read_csv(PROCESSED_DIR / "audio_features.csv").set_index("track_id")["bpm"].to_dict(),
        reqs=reqs,
        gt_cat=pd.read_csv(PROCESSED_DIR / "ground_truth_per_category.csv"),
        gt_track=pd.read_csv(PROCESSED_DIR / "ground_truth_per_track.csv"),
        records=pd.read_csv(PROCESSED_DIR / "mismatch_records.csv"),
        prompts=df["actual_prompt"],
        comments=df["Comments"],
    )


def subsets(cells: pd.DataFrame) -> dict:
    """(A) commented tracks = main result, (A-reviewed) without cells that still need review, (B) all."""
    a = cells[cells["gt_source"] != "no_comment_assumed_ok"]
    return {"A_commented": a, "A_reviewed": a[~a["gt_needs_review"]], "B_all": cells}


# ---------------------------------------------------------------------------
# evaluation sections
# ---------------------------------------------------------------------------

def evaluate_categories(preds, n_boot):
    """Per-category metrics for the modular system and the trivial baseline, on every GT subset."""
    frames = []
    for name, cells in subsets(preds).items():
        trivial = cells.assign(pred_mismatch=cells["n_requirements"] > 0, pred_score=cells["n_requirements"])
        for system, c in (("modular", cells), ("trivial", trivial)):
            frames.append(category_metrics(c, CATEGORIES, n_boot).assign(system=system, subset=name))
    return pd.concat(frames, ignore_index=True)


def evaluate_track_level(preds, gt_track, track_ids, E, prompts, embed_text, n_boot):
    """any_mismatch per track: modular system vs global CLAP similarity vs trivial baseline."""
    pos = {t: i for i, t in enumerate(track_ids)}
    per_track = preds.groupby("track_id").agg(modular=("pred_score", "max"), modular_any=("pred_mismatch", "any"),
                                              trivial=("n_requirements", lambda n: (n > 0).sum()))
    texts = sorted(set(prompts[track_ids]))
    text_emb = dict(zip(texts, embed_text(texts)))
    per_track["prompt_similarity"] = [float(E[pos[t]] @ text_emb[prompts[t]]) for t in per_track.index]
    t = gt_track.set_index("track_id").join(per_track)

    rows, roc = [], {}
    for name, part in (("A_commented", t[t["has_comment"]]), ("B_all", t)):
        y, groups = part["any_mismatch"].to_numpy(bool), part["original_id"].to_numpy()
        systems = {"modular system": part["modular"], "CLAP global similarity": -part["prompt_similarity"],
                   "trivial (n categories with a requirement)": part["trivial"]}
        for system, score in systems.items():
            pred, thresholds = oof_threshold_predictions(y, score.to_numpy(float), groups)
            rows.append(dict(subset=name, system=system, thresholds=", ".join(f"{x:.3g}" for x in thresholds),
                             **metrics_with_ci(y, pred, score, groups, n_boot)))
            if name == "B_all":
                roc[system] = roc_curve(y, score)
        rows.append(dict(subset=name, system="modular (own rule: any category flagged)", thresholds="",
                         **metrics_with_ci(y, part["modular_any"].to_numpy(bool), part["modular"], groups, n_boot)))
    return pd.DataFrame(rows), roc


def tau_sweep(verdicts, gt_cat, track_ids, n_boot):
    """F1 per category for tau 0.3-0.8 (CLAP checks only) on subset A, and tau chosen by GroupKFold."""
    a_keys = gt_cat[gt_cat["gt_source"] != "no_comment_assumed_ok"]
    enough = [c for c in CATEGORIES if a_keys.loc[a_keys["category"] == c, "gt_mismatch"].sum() >= MIN_POSITIVES]
    cells, by_tau, rows = None, {}, []
    for tau in TAUS:
        p = predictions_per_category(apply_tau(verdicts, tau), track_ids).merge(a_keys, on=["track_id", "category"])
        cells = p if cells is None else cells
        by_tau[tau] = p["pred_mismatch"].to_numpy(bool)
        y, cat = p["gt_mismatch"].to_numpy(bool), p["category"].to_numpy()
        row = {c: binary_metrics(y[cat == c], by_tau[tau][cat == c])["f1"] for c in enough}
        m = np.isin(cat, enough)
        rows.append(dict(tau=tau, **row, micro=binary_metrics(y[m], by_tau[tau][m])["f1"]))
    sweep = pd.DataFrame(rows)

    y, cat, groups = cells["gt_mismatch"].to_numpy(bool), cells["category"].to_numpy(), cells["original_id"]
    m = np.isin(cat, enough)
    oof, chosen = np.zeros(len(y), bool), []
    for train, test in GroupKFold(n_splits=5).split(np.zeros((len(y), 1)), y, groups):
        tr = train[m[train]]
        best = max(TAUS, key=lambda t: binary_metrics(y[tr], by_tau[t][tr])["f1"])
        chosen.append(float(best))
        oof[test] = by_tau[best][test]
    cv = category_metrics(cells.assign(pred_mismatch=oof), CATEGORIES, n_boot)
    cv["chosen_taus"] = ", ".join(f"{t:.2f}" for t in chosen)
    return sweep, cv, enough


def evaluate_vocals(evidence, gt_track, reqs, comments, n_boot):
    """CLAP vocal presence vs vocals_present_gt, per label source; disagreements go to a listening sheet."""
    v = gt_track.set_index("track_id")[["original_id", "vocals_present_gt", "vocals_gt_source"]].join(evidence["p_vocals"])
    v["y"] = v["vocals_present_gt"].astype(int)
    v["pred"] = (v["p_vocals"] >= 0.5).astype(int)
    rows = []
    for source, part in (("comment", v[v["vocals_gt_source"] == "comment"]),
                         ("instrumental_flag", v[v["vocals_gt_source"] == "instrumental_flag"]), ("all", v)):
        rows.append(dict(label_source=source, **multiclass_metrics_with_ci(part["y"], part["pred"], [0, 1],
                                                                          part["original_id"], n_boot)))
    vocal_reqs = reqs[(reqs["category"] == "vocals") & (reqs["value"] == "vocals")]
    says = vocal_reqs.groupby("track_id")["polarity"].min().map({1: "asks for vocals", -1: "no vocals"})
    disagree = v[v["y"] != v["pred"]].assign(prompt_says=lambda d: d.index.map(says).fillna("not mentioned"),
                                             expert_comment=lambda d: d.index.map(comments))
    sheet = disagree.reset_index()[["track_id", "vocals_gt_source", "vocals_present_gt", "p_vocals",
                                    "prompt_says", "expert_comment"]].round({"p_vocals": 3})
    sheet["my_label"] = ""  # vocals / no vocals, by listening
    path = RESULTS / "vocals_listening.csv"
    if path.exists():
        print(f"  {path.name} already exists, not overwriting it")
    else:
        sheet.to_csv(path, index=False)
    return pd.DataFrame(rows), len(sheet)


def evaluate_tempo_listening(verdicts, evidence, class_means):
    """Accuracy of the tempo verdicts against my_label in results/tempo_listening.csv (if filled)."""
    sheet = pd.read_csv(RESULTS / "tempo_listening.csv", dtype=str).fillna("")
    sheet["my_label"] = sheet["my_label"].str.strip().str.lower()
    sheet = sheet[sheet["my_label"].isin(TEMPO_CLASSES)]
    if sheet.empty:
        return None
    tempo = verdicts[verdicts["check"] == "tempo"].set_index("track_id")
    rows = []
    for _, r in sheet.iterrows():
        word = tempo.loc[r["track_id"], "value"]
        top = TEMPO_CLASSES[int(np.argmax(prior_correct(evidence.loc[r["track_id"], TEMPO_COLS].to_numpy(float),
                                                        class_means)))]
        truth = "satisfied" if r["my_label"] in TEMPO_WORD_CLASSES[word] else "changed"
        rows.append(dict(verdict_ok=tempo.loc[r["track_id"], "verdict"] == truth, class_ok=top == r["my_label"]))
    rows = pd.DataFrame(rows)
    return dict(n=len(rows), verdict_accuracy=rows["verdict_ok"].mean(), class_accuracy=rows["class_ok"].mean())


def evaluate_emotion(E, track_ids, gt_track, embed_text, logit_scale, n_boot):
    """Zero-shot over the 8 SunoCaps emotions vs emotion_gt, with a majority-class baseline."""
    gt = gt_track.set_index("track_id").loc[track_ids]
    P = softmax_probs(E, embed_text([f"{e} music" for e in SUNOCAPS_EMOTIONS]), logit_scale)
    labels = np.array(SUNOCAPS_EMOTIONS)
    majority = gt["emotion_gt"].value_counts().idxmax()
    preds = {"CLAP zero-shot": labels[P.argmax(1)],
             "CLAP zero-shot + prior correction": labels[(P / P.mean(0)).argmax(1)],
             f"majority class ({majority})": np.full(len(gt), majority)}
    rows = [dict(method=m, **multiclass_metrics_with_ci(gt["emotion_gt"], p, SUNOCAPS_EMOTIONS, gt["original_id"],
                                                        n_boot)) for m, p in preds.items()]
    confusion = pd.crosstab(pd.Categorical(gt["emotion_gt"], SUNOCAPS_EMOTIONS),
                            pd.Categorical(preds["CLAP zero-shot"], SUNOCAPS_EMOTIONS),
                            rownames=["expert emotion"], colnames=["CLAP zero-shot"], dropna=False)
    return pd.DataFrame(rows), confusion


def verdict_shares(verdicts, records, gt_track):
    """Share of satisfied / ignored / changed / hallucinated per category, model vs experts (commented tracks).

    Experts: ignored / changed / hallucinated = their mismatch records; satisfied = the checked
    requirements of that category minus the expert's ignored + changed records (an approximation,
    expert records are not matched one-to-one to parsed requirements).
    """
    commented = set(gt_track.loc[gt_track["has_comment"], "track_id"])
    v = verdicts[verdicts["track_id"].isin(commented)]
    model = pd.crosstab(v["category"], v["verdict"]).reindex(index=CATEGORIES, columns=SHOWN_VERDICTS, fill_value=0)
    checked = v["verdict"].isin(["satisfied", "ignored", "changed"])  # requirement rows with a real verdict
    n_req = v[checked].groupby("category").size().reindex(CATEGORIES, fill_value=0)
    rec = records[records["track_id"].isin(commented)]
    experts = pd.crosstab(rec["category"], rec["verdict"]).reindex(index=CATEGORIES, columns=SHOWN_VERDICTS,
                                                                   fill_value=0)
    experts["satisfied"] = (n_req - experts["ignored"] - experts["changed"]).clip(lower=0)
    rows = []
    for source, counts in (("model", model), ("experts", experts)):
        shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        for cat in CATEGORIES:
            rows.append(dict(source=source, category=cat, n=int(counts.loc[cat].sum()),
                             **{f"n_{k}": int(counts.loc[cat, k]) for k in SHOWN_VERDICTS},
                             **{f"share_{k}": shares.loc[cat, k] for k in SHOWN_VERDICTS}))
    return pd.DataFrame(rows)


def prompt_type_comparison(verdicts, gt_track, n_boot):
    """aspect_list vs caption, paired within original_id: share of satisfied requirements (model)
    and share of tracks with an expert mismatch (all 256 tracks, no comment = OK)."""
    v = verdicts[(verdicts["check"] != "vocal_hallucination") & verdicts["verdict"].isin(["satisfied", "ignored",
                                                                                           "changed"])]
    measures = {"model: share of satisfied requirements (all categories)": v}
    for cat in CATEGORIES:
        measures[f"model: share of satisfied requirements ({cat})"] = v[v["category"] == cat]
    rows = []
    for name, part in measures.items():
        per_track = part.assign(ok=part["verdict"] == "satisfied").groupby(
            ["original_id", "prompt_type", "track_id"])["ok"].mean()
        rows.append(paired_row(name, per_track.groupby(["original_id", "prompt_type"]).mean().unstack(), n_boot))
    expert = gt_track.groupby(["original_id", "prompt_type"])["any_mismatch"].mean().unstack()
    rows.append(paired_row("experts: share of tracks with a mismatch (no comment = OK)", expert, n_boot))
    return pd.DataFrame(rows)


def compare_systems(inp, track_ids, evidence, clap_scores, class_means, tau, essentia, n_boot):
    """CLAP vs Essentia vs their average with the same verdict rules, metrics and CIs (+ trivial baseline),
    and the paired difference Essentia - CLAP (same bootstrap resamples for both systems)."""
    ess_scores = essentia_percentiles(inp["reqs"], essentia, track_ids)
    ev_all = add_essentia_voice(evidence, essentia)
    gt = inp["gt_track"].set_index("track_id")
    per_cat, track, vocals, all_v = [], [], [], []
    kept = {}  # system -> (subset-A cells, {subset: (y, oof pred, score, groups)}, vocal predictions)
    for system in SYSTEMS:
        ev, scores = system_inputs(system, ev_all, clap_scores, ess_scores)
        v = all_verdicts(inp["reqs"], ev, scores, class_means, tau, inp["bpm"])
        all_v.append(v.assign(system=system))
        preds = predictions_per_category(v, track_ids).merge(inp["gt_cat"][GT_COLS], on=["track_id", "category"])
        parts = subsets(preds)
        for name in ("A_commented", "B_all"):
            per_cat.append(category_metrics(parts[name], CATEGORIES, n_boot).assign(system=system, subset=name))
            if system == "clap":  # the trivial baseline does not depend on the audio model
                cells = parts[name]
                trivial = cells.assign(pred_mismatch=cells["n_requirements"] > 0, pred_score=cells["n_requirements"])
                per_cat.append(category_metrics(trivial, CATEGORIES, n_boot).assign(system="trivial", subset=name))
        t = gt.join(preds.groupby("track_id")["pred_score"].max().rename("score"))
        track_parts = {}
        for name, part in (("A_commented", t[t["has_comment"]]), ("B_all", t)):
            y, groups, score = part["any_mismatch"].to_numpy(bool), part["original_id"].to_numpy(), part["score"]
            pred, _ = oof_threshold_predictions(y, score.to_numpy(float), groups)
            track.append(dict(system=system, subset=name, **metrics_with_ci(y, pred, score, groups, n_boot)))
            track_parts[name] = (y, pred, score.to_numpy(float), groups)
        vv = gt.join(ev["p_vocals"])
        vocal_pred = (vv["p_vocals"] >= 0.5).astype(int)
        vocals.append(dict(system=system, **multiclass_metrics_with_ci(
            vv["vocals_present_gt"].astype(int), vocal_pred, [0, 1], vv["original_id"], n_boot)))
        kept[system] = (parts["A_commented"], track_parts, vocal_pred.to_numpy())

    # paired differences Essentia - CLAP
    paired = []
    cells_c, cells_e = kept["clap"][0], kept["essentia"][0]
    cat = cells_c["category"].to_numpy()
    y = cells_c["gt_mismatch"].to_numpy(bool)
    enough = [c for c in CATEGORIES if y[cat == c].sum() >= MIN_POSITIVES]
    for c in enough + ["micro"]:
        m = np.isin(cat, enough) if c == "micro" else cat == c
        diff = paired_difference(y[m], cells_c["pred_mismatch"].to_numpy()[m], cells_c["pred_score"].to_numpy()[m],
                                 cells_e["pred_mismatch"].to_numpy()[m], cells_e["pred_score"].to_numpy()[m],
                                 cells_c["original_id"].to_numpy()[m], n_boot)
        paired.append(dict(level="category", subset="A_commented", category=c, **diff))
    for name in ("A_commented", "B_all"):
        (y_t, pred_c, score_c, groups), (_, pred_e, score_e, _) = kept["clap"][1][name], kept["essentia"][1][name]
        paired.append(dict(level="track", subset=name, category="any mismatch",
                           **paired_difference(y_t, pred_c, score_c, pred_e, score_e, groups, n_boot)))
    y_v = gt["vocals_present_gt"].astype(int).to_numpy()
    acc = lambda pred, i: float(np.mean(pred[i] == y_v[i]))  # noqa: E731
    boot = cluster_bootstrap(lambda i: {"accuracy_diff": acc(kept["essentia"][2], i) - acc(kept["clap"][2], i)},
                             gt["original_id"].to_numpy(), n_boot)
    point = {"accuracy_diff": acc(kept["essentia"][2], np.arange(len(y_v))) - acc(kept["clap"][2], np.arange(len(y_v)))}
    paired.append(dict(level="vocal presence", subset="all", category="accuracy",
                       **with_ci(point, boot, ["accuracy_diff"])))

    n_ess = int(inp["reqs"]["check"].isin(ESSENTIA_PREFIX).sum())
    info = dict(tracks=len(essentia), missing_tracks=len(set(track_ids) - set(essentia.index)),
                essentia_requirements=n_ess, with_essentia_score=int(ess_scores["score_pct"].notna().sum()),
                with_voice_score=int(ev_all["p_voice"].notna().sum()))
    return (pd.concat(per_cat, ignore_index=True), pd.DataFrame(track), pd.DataFrame(vocals), pd.DataFrame(paired),
            info, pd.concat(all_v))


def paired_row(name, wide, n_boot):
    wide = wide.dropna()
    diff = (wide["aspect_list"] - wide["caption"]).to_numpy()
    boot = cluster_bootstrap(lambda i: {"diff": diff[i].mean()}, wide.index.to_numpy(), n_boot)
    p = wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan
    return dict(measure=name, n_pairs=len(diff), aspect_list=wide["aspect_list"].mean(), caption=wide["caption"].mean(),
                **with_ci({"diff": diff.mean()}, boot, ["diff"]), wilcoxon_p=p)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


def plot_tau_sweep(sweep, cv, enough, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _axes(ax)
    for color, c in zip(SLOTS, enough):
        ax.plot(sweep["tau"], sweep[c], color=color, linewidth=2, marker="o", markersize=4, label=c)
    ax.plot(sweep["tau"], sweep["micro"], color=INK, linewidth=2, linestyle="--", label="micro (these 4)")
    ax.axvline(DEFAULT_TAU, color=MUTED, linewidth=1, linestyle=":")
    ax.text(DEFAULT_TAU + 0.005, ax.get_ylim()[1], " default tau", color=MUTED, va="top", fontsize=8)
    ax.set_xlabel("tau (CLAP checks: satisfied if score percentile >= tau)", color=MUTED)
    ax.set_ylabel("F1 (subset A, commented tracks)", color=MUTED)
    ax.set_title(f"F1 per category vs tau  (GroupKFold picks: {cv['chosen_taus'].iloc[0]})", loc="left",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_verdict_shares(shares, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE, sharey=True)
    for ax, source in zip(axes, ("model", "experts")):
        _axes(ax)
        ax.grid(False)
        part = shares[shares["source"] == source].set_index("category").loc[CATEGORIES[::-1]]
        left = np.zeros(len(part))
        for color, k in zip(SLOTS, SHOWN_VERDICTS):
            vals = part[f"share_{k}"].fillna(0).to_numpy()
            ax.barh(part.index, vals, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=k, height=0.6)
            left += vals
        ax.set_xlim(0, 1)
        ax.set_title(f"{source} (commented tracks)", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("share of verdicts", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emotion_confusion(confusion, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    m = confusion.to_numpy()
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=45, ha="right", color=MUTED)
    ax.set_yticks(range(len(confusion.index)), confusion.index, color=MUTED)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > m.max() / 2 else INK)
    ax.set_xlabel("CLAP zero-shot", color=MUTED)
    ax.set_ylabel("expert emotion", color=MUTED)
    ax.set_title("Emotion: expert label vs CLAP zero-shot (256 tracks)", loc="left", color=INK, fontsize=10)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_track_roc(roc, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    _axes(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for color, (system, (fpr, tpr, _)) in zip(SLOTS, roc.items()):
        ax.plot(fpr, tpr, color=color, linewidth=2, label=system)
    ax.set_xlabel("false positive rate", color=MUTED)
    ax.set_ylabel("true positive rate", color=MUTED)
    ax.set_title("Track-level any mismatch (subset B, 256 tracks)", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------

def md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def write_summary(path, n_boot, tau, per_cat, track, sweep_cv, enough, vocals, n_vocal_sheet, tempo_listen,
                  emotion, shares, ptype, verdicts, systems=None):
    out = ["# Results summary", "",
           f"Model `laion/larger_clap_general` (cached embeddings), tau = {tau}, 95% CIs from a cluster bootstrap "
           f"over original_id ({n_boot} resamples). Subset A = commented tracks (main result), "
           "A-reviewed = A without cells that still need review, B = all 256 tracks with no comment = OK. "
           f"Categories with fewer than {MIN_POSITIVES} expert positives are marked *not enough labels*.", "",
           "Scores for ROC-AUC (`pred_score` = max over the requirements of a category): 1 - score percentile for "
           "CLAP checks, the CLAP probability of the wrong answer for vocal presence / gender / hallucination, "
           "0/1 for the rule-based tempo check. Essentia is postponed, so `essentia_*` requirements use CLAP.", ""]

    counts = verdicts["verdict"].value_counts()
    out += ["## Verdicts", "", f"{len(verdicts)} rows: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in
            SHOWN_VERDICTS + ["not_checked", "none", "unknown"]) + " (none = no hallucinated vocals).", ""]

    out += ["## 1. Per category, subset A (main result)", ""]
    a = per_cat[(per_cat["subset"] == "A_commented")]
    rows = []
    for _, r in a[a["system"] == "modular"].iterrows():
        triv = a[(a["system"] == "trivial") & (a["category"] == r["category"])].iloc[0]
        rows.append([r["category"], r["support"], r["n"], fmt_ci(r, "precision"), fmt_ci(r, "recall"),
                     fmt_ci(r, "f1"), fmt_ci(r, "roc_auc"), fmt_ci(triv, "f1"), r["note"]])
    out += [md_table(["category", "positives", "cells", "precision", "recall", "F1", "ROC-AUC", "trivial F1", "note"],
                     rows), "",
            "Experts never flagged tempo or mood, so every model flag there counts as a false positive; those rows "
            "are not an evaluation of the tempo / mood checks.", ""]

    out += ["## 2. Micro / macro across GT subsets", ""]
    rows = []
    for _, r in per_cat[per_cat["category"].isin(["micro", "macro"])].iterrows():
        rows.append([r["subset"], r["system"], r["category"], fmt_ci(r, "precision"), fmt_ci(r, "recall"),
                     fmt_ci(r, "f1"), fmt_ci(r, "roc_auc")])
    out += [md_table(["subset", "system", "average", "precision", "recall", "F1", "ROC-AUC"], rows),
            "", f"Micro and macro use the categories with enough labels ({', '.join(enough)}).", ""]

    out += ["## 3. Track level: any mismatch", ""]
    rows = [[r["subset"], r["system"], f"{r['support']}/{r['n']}", fmt_ci(r, "roc_auc"), fmt_ci(r, "f1"),
             fmt_ci(r, "precision"), fmt_ci(r, "recall")] for _, r in track.iterrows()]
    out += [md_table(["subset", "system", "positives", "ROC-AUC", "F1", "precision", "recall"], rows), "",
            "F1 uses a threshold chosen on the other folds (5-fold GroupKFold over original_id). Subset A has very "
            "few negatives, so its ROC-AUC is noisy; B is the more informative track-level comparison.", ""]

    out += ["## 4. tau for the CLAP checks", "", f"GroupKFold picks tau = {sweep_cv['chosen_taus'].iloc[0]} "
            "(one per fold). Out-of-fold F1 with the chosen tau vs the default:", ""]
    a_mod = a[a["system"] == "modular"].set_index("category")
    rows = [[c, fmt_ci(a_mod.loc[c], "f1"), fmt_ci(sweep_cv.set_index("category").loc[c], "f1")]
            for c in enough + ["micro", "macro"]]
    out += [md_table(["category", f"F1, tau = {tau}", "F1, tau by GroupKFold"], rows), "",
            "![tau sweep](figures/tau_sweep.png)", ""]

    out += ["## 5. Vocal presence (CLAP vs vocals_present_gt)", ""]
    rows = [[r["label_source"], r["n"], fmt_ci(r, "accuracy"), fmt_ci(r, "macro_f1")] for _, r in vocals.iterrows()]
    out += [md_table(["label source", "tracks", "accuracy", "macro-F1"], rows), "",
            f"{n_vocal_sheet} disagreements are in `results/vocals_listening.csv` for listening.", ""]

    out += ["## 6. Tempo (listening check)", ""]
    if tempo_listen is None:
        out += ["`results/tempo_listening.csv` is not labelled yet, so there is no tempo accuracy.", ""]
    else:
        out += [f"{tempo_listen['n']} labelled tracks: verdict accuracy {tempo_listen['verdict_accuracy']:.2f}, "
                f"tempo class accuracy {tempo_listen['class_accuracy']:.2f}.", ""]

    out += ["## 7. Emotion (8 SunoCaps emotions, zero-shot)", ""]
    rows = [[r["method"], fmt_ci(r, "accuracy"), fmt_ci(r, "macro_f1")] for _, r in emotion.iterrows()]
    out += [md_table(["method", "accuracy", "macro-F1"], rows), "", "![emotion](figures/emotion_confusion.png)", ""]

    out += ["## 8. Verdict shares per category (commented tracks)", ""]
    rows = [[r["source"], r["category"], r["n"]] + [f"{r[f'share_{k}']:.2f}" if pd.notna(r[f'share_{k}']) else "–"
                                                   for k in SHOWN_VERDICTS] for _, r in shares.iterrows()]
    out += [md_table(["source", "category", "n"] + SHOWN_VERDICTS, rows), "",
            "Expert *satisfied* = parsed requirements minus expert ignored/changed records (approximation).", "",
            "![verdict shares](figures/verdict_shares.png)", ""]

    out += ["## 9. Aspect list vs caption (paired within original_id)", ""]
    rows = [[r["measure"], r["n_pairs"], f"{r['aspect_list']:.3f}", f"{r['caption']:.3f}",
             fmt_ci(r, "diff", 3), f"{r['wilcoxon_p']:.3f}" if pd.notna(r["wilcoxon_p"]) else "–"]
            for _, r in ptype.iterrows()]
    out += [md_table(["measure", "pairs", "aspect list", "caption", "difference (aspect - caption)", "Wilcoxon p"],
                     rows), "", "Rates, not counts: aspect prompts have ~10.5 requirements, captions ~6.5.", ""]

    out += ["## 10. CLAP vs Essentia vs their average", ""]
    if systems is None:
        out += [f"`data/processed/{ESSENTIA_CSV.name}` not found, so this comparison is skipped. Run "
                "`notebooks/04_essentia.ipynb` in Colab and copy the file to `data/processed/`.", ""]
    else:
        sys_cat, sys_track, sys_vocals, sys_paired, info = systems
        out += [f"Essentia scores for {info['tracks']} tracks ({info['missing_tracks']} missing). "
                f"{info['with_essentia_score']} of {info['essentia_requirements']} `essentia_*` requirements got an "
                "Essentia score (the rest keep CLAP: label not in the Essentia models or track missing); "
                f"{info['with_voice_score']} tracks have an Essentia voice score. Same verdict rules, tau and "
                "calibration for every system; subset A (commented tracks).", ""]
        a = sys_cat[sys_cat["subset"] == "A_commented"]
        rows = []
        for c in enough + ["micro", "macro"]:
            r = {s: a[(a["system"] == s) & (a["category"] == c)].iloc[0] for s in SYSTEMS + ["trivial"]}
            rows.append([c, r["clap"]["support"]] + [fmt_ci(r[s], "f1") for s in SYSTEMS + ["trivial"]]
                        + [fmt_ci(r[s], "roc_auc") for s in SYSTEMS])
        out += [md_table(["category", "positives", "F1 CLAP", "F1 Essentia", "F1 average", "F1 trivial",
                          "AUC CLAP", "AUC Essentia", "AUC average"], rows), ""]
        rows = [[r["subset"], r["system"], fmt_ci(r, "roc_auc"), fmt_ci(r, "f1")] for _, r in sys_track.iterrows()]
        out += ["Track level (any mismatch):", "", md_table(["subset", "system", "ROC-AUC", "F1"], rows), ""]
        rows = [[r["system"], fmt_ci(r, "accuracy"), fmt_ci(r, "macro_f1")] for _, r in sys_vocals.iterrows()]
        out += ["Vocal presence vs `vocals_present_gt` (all 256 tracks):", "",
                md_table(["system", "accuracy", "macro-F1"], rows), ""]
        rows = []
        for _, r in sys_paired.iterrows():
            if r["level"] == "vocal presence":
                rows.append([r["level"], r["subset"], "accuracy", fmt_ci(r, "accuracy_diff"), ""])
            else:
                rows.append([r["level"], r["subset"], r["category"], fmt_ci(r, "f1_diff"), fmt_ci(r, "roc_auc_diff")])
        out += ["Paired difference **Essentia − CLAP** (same bootstrap resamples for both; a CI that excludes 0 "
                "means a real difference):", "",
                md_table(["level", "subset", "category", "Δ F1 (or Δ accuracy)", "Δ ROC-AUC"], rows), ""]
    path.write_text("\n".join(out))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    inp = load_inputs()
    scorer = ClapScorer(inp["meta"]["model"])  # text embeddings only
    logit_scale = inp["meta"]["logit_scale"]
    track_ids, E = stack_track_embeddings(inp["window_embs"])

    print("verdicts ...", flush=True)
    evidence = audio_evidence(track_ids, E, scorer.embed_text, logit_scale)
    class_means = evidence[TEMPO_COLS].mean().to_numpy()
    clap_scores = clap_percentiles(inp["reqs"], track_ids, E, scorer.embed_text)
    verdicts = all_verdicts(inp["reqs"], evidence, clap_scores, class_means, args.tau, inp["bpm"])
    verdicts.assign(params=verdicts["params"].map(lambda p: json.dumps(p) if isinstance(p, dict) else "")).to_csv(
        PROCESSED_DIR / "requirement_verdicts.csv", index=False)
    preds = predictions_per_category(verdicts, track_ids).merge(inp["gt_cat"][GT_COLS], on=["track_id", "category"])
    preds.to_csv(PROCESSED_DIR / "predictions_per_category.csv", index=False)
    print(verdicts["verdict"].value_counts().to_string())

    print("per-category metrics ...", flush=True)
    per_cat = evaluate_categories(preds, args.n_boot)
    per_cat.to_csv(RESULTS / "metrics_per_category.csv", index=False)

    print("track level ...", flush=True)
    track, roc = evaluate_track_level(preds, inp["gt_track"], track_ids, E, inp["prompts"], scorer.embed_text,
                                      args.n_boot)
    track.to_csv(RESULTS / "metrics_track_level.csv", index=False)
    plot_track_roc(roc, FIGURES / "track_level_roc.png")

    print("tau sweep ...", flush=True)
    sweep, sweep_cv, enough = tau_sweep(verdicts, inp["gt_cat"], track_ids, args.n_boot)
    sweep.to_csv(RESULTS / "metrics_tau_sweep.csv", index=False)
    sweep_cv.to_csv(RESULTS / "metrics_tau_cv.csv", index=False)
    plot_tau_sweep(sweep, sweep_cv, enough, FIGURES / "tau_sweep.png")

    print("vocals, tempo, emotion ...", flush=True)
    vocals, n_vocal_sheet = evaluate_vocals(evidence, inp["gt_track"], inp["reqs"], inp["comments"], args.n_boot)
    vocals.to_csv(RESULTS / "metrics_vocals.csv", index=False)
    tempo_listen = evaluate_tempo_listening(verdicts, evidence, class_means)
    emotion, confusion = evaluate_emotion(E, track_ids, inp["gt_track"], scorer.embed_text, logit_scale, args.n_boot)
    emotion.to_csv(RESULTS / "metrics_emotion.csv", index=False)
    confusion.to_csv(RESULTS / "metrics_emotion_confusion.csv")
    plot_emotion_confusion(confusion, FIGURES / "emotion_confusion.png")

    print("analysis ...", flush=True)
    shares = verdict_shares(verdicts, inp["records"], inp["gt_track"])
    shares.to_csv(RESULTS / "metrics_verdict_shares.csv", index=False)
    plot_verdict_shares(shares, FIGURES / "verdict_shares.png")
    ptype = prompt_type_comparison(verdicts, inp["gt_track"], args.n_boot)
    ptype.to_csv(RESULTS / "metrics_prompt_type.csv", index=False)

    essentia, systems = load_essentia_scores(), None
    if essentia is None:
        print(f"no {ESSENTIA_CSV.name} in data/processed/ - skipping the Essentia comparison "
              "(run notebooks/04_essentia.ipynb in Colab first)")
    else:
        print(f"CLAP vs Essentia vs average ({len(essentia)} tracks with Essentia scores) ...", flush=True)
        sys_cat, sys_track, sys_vocals, sys_paired, info, sys_verdicts = compare_systems(
            inp, track_ids, evidence, clap_scores, class_means, args.tau, essentia, args.n_boot)
        sys_cat.to_csv(RESULTS / "metrics_systems_per_category.csv", index=False)
        sys_track.to_csv(RESULTS / "metrics_systems_track_level.csv", index=False)
        sys_vocals.to_csv(RESULTS / "metrics_systems_vocals.csv", index=False)
        sys_paired.to_csv(RESULTS / "metrics_systems_paired.csv", index=False)
        sys_verdicts.drop(columns=["params"]).to_csv(PROCESSED_DIR / "requirement_verdicts_systems.csv", index=False)
        systems = (sys_cat, sys_track, sys_vocals, sys_paired, info)

    write_summary(RESULTS / "summary.md", args.n_boot, args.tau, per_cat, track, sweep_cv, enough, vocals,
                  n_vocal_sheet, tempo_listen, emotion, shares, ptype, verdicts, systems)
    print(f"done -> {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
