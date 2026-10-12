"""
test_evaluate.py — checks for the from-scratch metrics in src/evaluate.py.

Three kinds of test:
  1. small examples whose answer is worked out by hand in the comment,
  2. agreement with scikit-learn's metrics on seeded random data,
  3. edge cases (undefined ratios, ties, single-class input).

scikit-learn is used here ONLY to check our numbers.

Run from the repo root:
    python -m pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from src import evaluate

SEED = 42
TOL = 1e-12


# --- 1. Hand-checked examples ----------------------------------------------
def test_regression_metrics_by_hand():
    y_true = [3.0, 5.0, 2.0, 7.0]
    y_pred = [2.0, 5.0, 4.0, 8.0]
    # errors = 1, 0, -2, -1 -> squared = 1, 0, 4, 1 -> mean = 1.5
    assert evaluate.rmse(y_true, y_pred) == pytest.approx(np.sqrt(1.5))
    # |errors| = 1, 0, 2, 1 -> mean = 1.0
    assert evaluate.mae(y_true, y_pred) == pytest.approx(1.0)
    # mean(y_true) = 4.25; SS_tot = 1.5625 + 0.5625 + 5.0625 + 7.5625 = 14.75
    # SS_res = 6 -> R2 = 1 - 6 / 14.75
    assert evaluate.r2(y_true, y_pred) == pytest.approx(1 - 6 / 14.75)


def test_classification_metrics_by_hand():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    # TP = 2, FN = 1, FP = 1, TN = 2
    cm = evaluate.confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[2, 1], [1, 2]]
    assert evaluate.accuracy(y_true, y_pred) == pytest.approx(4 / 6)
    precision, recall, f1 = evaluate.precision_recall_f1(y_true, y_pred)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_roc_auc_by_hand():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    # (positive, negative) pairs: (0.35, 0.1) win, (0.35, 0.4) loss,
    # (0.8, 0.1) win, (0.8, 0.4) win -> 3 of 4 pairs ranked correctly.
    assert evaluate.roc_auc(y_true, scores) == pytest.approx(0.75)


def test_roc_auc_tie_counts_as_half():
    y_true = [0, 1]
    scores = [0.5, 0.5]
    assert evaluate.roc_auc(y_true, scores) == pytest.approx(0.5)


# --- 2. Agreement with scikit-learn ----------------------------------------
@pytest.mark.parametrize("n", [10, 200, 5000])
def test_regression_matches_sklearn(n):
    rng = np.random.default_rng(SEED + n)
    y_true = rng.lognormal(mean=12.0, sigma=0.8, size=n)   # price-like, heavy tail
    y_pred = y_true * rng.normal(loc=1.0, scale=0.3, size=n)

    assert evaluate.rmse(y_true, y_pred) == pytest.approx(
        np.sqrt(skm.mean_squared_error(y_true, y_pred)), rel=TOL
    )
    assert evaluate.mae(y_true, y_pred) == pytest.approx(
        skm.mean_absolute_error(y_true, y_pred), rel=TOL
    )
    assert evaluate.r2(y_true, y_pred) == pytest.approx(
        skm.r2_score(y_true, y_pred), rel=TOL
    )


@pytest.mark.parametrize("n", [10, 200, 5000])
@pytest.mark.parametrize("positive_rate", [0.5, 0.1])
def test_classification_matches_sklearn(n, positive_rate):
    rng = np.random.default_rng(SEED + n)
    y_true = (rng.random(n) < positive_rate).astype(int)
    y_true[:2] = [0, 1]                                    # both classes present
    flip = rng.random(n) < 0.2                             # 20% wrong predictions
    y_pred = np.where(flip, 1 - y_true, y_true)

    assert evaluate.confusion_matrix(y_true, y_pred).tolist() == (
        skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    )
    assert evaluate.accuracy(y_true, y_pred) == pytest.approx(
        skm.accuracy_score(y_true, y_pred), rel=TOL
    )
    precision, recall, f1 = evaluate.precision_recall_f1(y_true, y_pred)
    assert precision == pytest.approx(
        skm.precision_score(y_true, y_pred, zero_division=0), rel=TOL
    )
    assert recall == pytest.approx(
        skm.recall_score(y_true, y_pred, zero_division=0), rel=TOL
    )
    assert f1 == pytest.approx(skm.f1_score(y_true, y_pred, zero_division=0), rel=TOL)


@pytest.mark.parametrize("n", [10, 200, 5000])
@pytest.mark.parametrize("positive_rate", [0.5, 0.1])
def test_roc_auc_matches_sklearn_continuous_scores(n, positive_rate):
    rng = np.random.default_rng(SEED + n)
    y_true = (rng.random(n) < positive_rate).astype(int)
    y_true[:2] = [0, 1]
    scores = y_true + rng.normal(scale=1.0, size=n)        # SVM-margin-like

    assert evaluate.roc_auc(y_true, scores) == pytest.approx(
        skm.roc_auc_score(y_true, scores), rel=TOL
    )


@pytest.mark.parametrize("n", [10, 200, 5000])
def test_roc_auc_matches_sklearn_with_heavy_ties(n):
    rng = np.random.default_rng(SEED + n)
    y_true = (rng.random(n) < 0.5).astype(int)
    y_true[:2] = [0, 1]
    # Only 5 distinct scores, like a shallow tree's leaf probabilities.
    scores = np.round(0.25 * y_true + rng.random(n) * 0.75, 1).clip(0, 0.4)

    assert evaluate.roc_auc(y_true, scores) == pytest.approx(
        skm.roc_auc_score(y_true, scores), rel=TOL
    )


# --- 3. Edge cases ----------------------------------------------------------
def test_perfect_regression():
    y = [1.0, 2.0, 3.0]
    assert evaluate.rmse(y, y) == 0.0
    assert evaluate.mae(y, y) == 0.0
    assert evaluate.r2(y, y) == 1.0


def test_r2_constant_target():
    y_true = [5.0, 5.0, 5.0]
    assert evaluate.r2(y_true, [5.0, 5.0, 5.0]) == 1.0
    assert evaluate.r2(y_true, [4.0, 5.0, 6.0]) == 0.0
    assert evaluate.r2(y_true, [4.0, 5.0, 6.0]) == skm.r2_score(y_true, [4.0, 5.0, 6.0])


def test_no_predicted_positives_gives_zero_precision():
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    assert evaluate.precision_recall_f1(y_true, y_pred) == (0.0, 0.0, 0.0)


def test_no_actual_positives_gives_zero_recall():
    y_true = [0, 0, 0, 0]
    y_pred = [1, 0, 0, 0]
    assert evaluate.precision_recall_f1(y_true, y_pred) == (0.0, 0.0, 0.0)


def test_roc_auc_perfect_and_inverted():
    y_true = [0, 0, 1, 1]
    assert evaluate.roc_auc(y_true, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert evaluate.roc_auc(y_true, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_roc_auc_needs_both_classes():
    with pytest.raises(ValueError):
        evaluate.roc_auc([1, 1, 1], [0.2, 0.5, 0.9])


def test_roc_curve_by_hand():
    # Scores high to low: 0.8 (premium), 0.4 (standard), 0.35 (premium), 0.1 (standard).
    # Lowering the threshold past each one: TP = 1, 1, 2, 2 and FP = 0, 1, 1, 2.
    fpr, tpr = evaluate.roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert fpr.tolist() == [0.0, 0.0, 0.5, 0.5, 1.0]
    assert tpr.tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]


@pytest.mark.parametrize("n_distinct", [5, 1000])
def test_roc_curve_matches_sklearn_and_its_area_is_roc_auc(n_distinct):
    rng = np.random.default_rng(n_distinct)
    y_true = rng.integers(0, 2, size=400)
    # Few distinct scores means many ties, as a shallow tree produces.
    noisy = rng.random(400) * 0.7 + y_true * 0.3               # premium tends to score higher
    scores = np.floor(noisy * n_distinct) / n_distinct

    fpr, tpr = evaluate.roc_curve(y_true, scores)
    sk_fpr, sk_tpr, _ = skm.roc_curve(y_true, scores, drop_intermediate=False)
    assert np.allclose(fpr, sk_fpr) and np.allclose(tpr, sk_tpr)

    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)      # trapezoids under the curve
    assert area == pytest.approx(evaluate.roc_auc(y_true, scores), rel=1e-12)


def test_roc_curve_needs_both_classes():
    with pytest.raises(ValueError):
        evaluate.roc_curve([0, 0, 0], [0.2, 0.5, 0.9])


def test_metrics_return_plain_floats():
    assert type(evaluate.rmse([1, 2], [1, 3])) is float
    assert type(evaluate.roc_auc([0, 1], [0.2, 0.9])) is float
    assert all(type(v) is float for v in evaluate.precision_recall_f1([0, 1], [0, 1]))


# --- Bundled dicts and the comparison table ---------------------------------
def test_metric_dicts():
    reg = evaluate.regression_metrics([3.0, 5.0], [2.0, 5.0])
    assert list(reg) == ["RMSE", "MAE", "R2"]

    y_true, y_pred = [1, 0, 1, 0], [1, 0, 0, 0]
    without_scores = evaluate.classification_metrics(y_true, y_pred)
    assert list(without_scores) == ["Accuracy", "Precision", "Recall", "F1"]

    with_scores = evaluate.classification_metrics(y_true, y_pred, [0.9, 0.2, 0.4, 0.1])
    assert with_scores["ROC-AUC"] == 1.0


def test_compare_prints_table(capsys):
    evaluate.compare(
        {
            "Ours: tree": {"RMSE": 81234.7, "F1": (0.9123, 0.0041)},
            "Ours: SVM": {"F1": 0.8857},
        }
    )
    lines = capsys.readouterr().out.splitlines()

    assert lines[0].split() == ["Model", "RMSE", "F1"]
    assert set(lines[1]) == {"-", " "}
    assert "81,235" in lines[2] and "0.9123 ± 0.0041" in lines[2]
    assert "n/a" in lines[3] and "0.8857" in lines[3]
    assert len({len(line) for line in lines}) == 1         # columns line up
