"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.

Conventions:
  - Task A (regression): y_true / y_pred are real numbers (price or log-price).
  - Task B (classification): labels are 0/1 with 1 = premium (the positive
    class), exactly as produced by data_prep.make_tier_label.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    """Root mean squared error: sqrt(mean((y_true - y_pred)^2))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error: mean(|y_true - y_pred|)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """
    Coefficient of determination: 1 - SS_res / SS_tot.
    1.0 is a perfect fit, 0.0 is no better than always predicting the mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        # Constant target: the ratio is undefined, so follow sklearn's rule.
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred) -> dict:
    """All Task A metrics in one dict, ready for compare()."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy(y_true, y_pred) -> float:
    """Share of correct predictions: (TP + TN) / n."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    # A zero denominator means the ratio is undefined; report 0.0 (as sklearn does).
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.

    AUC is the probability that a random premium listing gets a higher score
    than a random standard one (a tie counts as half):

        AUC = (R_pos - n_pos (n_pos + 1) / 2) / (n_pos * n_neg)

    where R_pos is the sum of the ranks of the positive examples.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC needs both classes present in y_true.")

    # Rank the scores from 1 (lowest) to n. Tied scores share the AVERAGE of
    # the ranks they occupy; tree probabilities come from leaf frequencies,
    # so ties are common.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)                  # highest rank in each tie group
    avg_rank = last_rank - (counts - 1) / 2.0      # middle rank of each tie group
    ranks = avg_rank[inverse]

    rank_sum_pos = np.sum(ranks[y_true == 1])
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(y_true, scores):
    """
    Points of the ROC curve, for plotting. Returns (fpr, tpr).

    Slide a threshold from the highest score down. At each distinct score,
    everything scoring at least that much is called premium, and we record
        fpr = FP / n_neg   (standard listings wrongly called premium)
        tpr = TP / n_pos   (premium listings found; this is recall)
    The curve starts at (0, 0) and ends at (1, 1). The area under it is roc_auc.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("The ROC curve needs both classes present in y_true.")

    order = np.argsort(-scores, kind="stable")         # highest score first
    sorted_scores = scores[order]
    is_pos = y_true[order] == 1
    tp = np.cumsum(is_pos)                             # premium listings above the threshold
    fp = np.cumsum(~is_pos)                            # standard listings above the threshold

    # Tied scores cross the threshold together: keep one point per distinct score.
    last_of_tie = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    fpr = np.concatenate([[0.0], fp[last_of_tie] / n_neg])
    tpr = np.concatenate([[0.0], tp[last_of_tie] / n_pos])
    return fpr, tpr


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    """
    All Task B metrics in one dict, ready for compare().
    ROC-AUC is included only when decision scores are given.
    """
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    metrics = {
        "Accuracy": accuracy(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }
    if scores is not None:
        metrics["ROC-AUC"] = roc_auc(y_true, scores)
    return metrics


# --- Comparison helper ------------------------------------------------------
def _format_value(value) -> str:
    """Format one table cell: a number, a (mean, std) pair, or a missing value."""
    if value is None:
        return "n/a"
    if isinstance(value, (tuple, list)):
        mean, std = value
        return f"{_format_number(mean)} ± {_format_number(std)}"
    return _format_number(value)


def _format_number(x) -> str:
    """4 decimals for scores in [-1, 1]-ish ranges, thousands separators for prices."""
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.4f}"


def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.

    Input:  {"model name": {"metric name": value, ...}, ...}
    A value is a float or a (mean, std) pair, printed as "mean ± std".
    A metric that a model does not have is printed as "n/a".
    """
    # Columns: every metric that appears, in first-seen order.
    columns: list[str] = []
    for metrics in name_to_metrics.values():
        for key in metrics:
            if key not in columns:
                columns.append(key)

    header = ["Model"] + columns
    rows = [
        [name] + [_format_value(metrics.get(col)) for col in columns]
        for name, metrics in name_to_metrics.items()
    ]

    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def render(row: list[str]) -> str:
        # Model names left-aligned, numbers right-aligned.
        cells = [row[0].ljust(widths[0])]
        cells += [cell.rjust(widths[i]) for i, cell in enumerate(row) if i > 0]
        return "  ".join(cells)

    print(render(header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(render(row))
