"""evaluation.py: metrics, cluster bootstrap, out-of-fold thresholds."""
import numpy as np

from aimv.evaluation import (binary_metrics, cluster_bootstrap, metrics_with_ci, oof_threshold_predictions,
                             paired_difference)


def test_binary_metrics():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0], score=[0.9, 0.4, 0.6, 0.1])
    assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5)
    assert m["roc_auc"] == 0.75 and m["support"] == 2 and m["n"] == 4


def test_auc_undefined_with_one_class():
    assert np.isnan(binary_metrics([0, 0], [1, 0], score=[0.5, 0.1])["roc_auc"])


def test_cluster_bootstrap_keeps_groups_together():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    sizes = cluster_bootstrap(lambda i: {"n": len(i), "pairs": np.all(groups[i][::2] == groups[i][1::2])},
                              groups, n_boot=50)
    assert (sizes["n"] == 6).all() and sizes["pairs"].all()  # always 3 whole groups of 2 rows


def test_metrics_with_ci_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = y + rng.normal(0, 0.8, 200)
    m = metrics_with_ci(y, score > 0.5, score, groups=np.arange(200) // 4, n_boot=200)
    assert m["roc_auc_lo"] <= m["roc_auc"] <= m["roc_auc_hi"]


def test_oof_threshold_predictions():
    y = np.array([0, 1] * 10)
    score = y * 0.8 + 0.1
    pred, thresholds = oof_threshold_predictions(y, score, groups=np.arange(20) // 2)
    assert len(thresholds) == 5 and (pred == y.astype(bool)).all()


def test_paired_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(bool)
    noisy = y + rng.normal(0, 1.0, 200)  # weak system
    sharp = y + rng.normal(0, 0.2, 200)  # strong system
    groups = np.arange(200) // 4
    same = paired_difference(y, noisy > 0.5, noisy, noisy > 0.5, noisy, groups, n_boot=100)
    assert same["f1_diff"] == 0 and same["roc_auc_diff_lo"] == 0 and same["roc_auc_diff_hi"] == 0
    better = paired_difference(y, noisy > 0.5, noisy, sharp > 0.5, sharp, groups, n_boot=200)
    assert better["roc_auc_diff"] > 0 and better["roc_auc_diff_lo"] > 0  # b - a, clearly above 0
