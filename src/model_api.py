"""
model_api.py — the one place that reads OPTIONAL outputs from a fitted model.

The pipeline works with our own tree and SVM and with the scikit-learn
baselines. Each function returns None when a model does not offer that output,
so a missing piece skips one figure instead of stopping the whole run.

If the attribute names in decision_tree.py or svm.py change, only this file
needs to change.
"""

from __future__ import annotations

import numpy as np

from . import benchmark


def _first_attribute(model, names):
    """The first of the named attributes the model has, as a float array, else None."""
    for name in names:
        value = getattr(model, name, None)
        if value is not None and not callable(value):
            return np.asarray(value, dtype=float)
    return None


def feature_importances(tree):
    """Impurity-decrease importance of each feature (one number per column), or None."""
    return _first_attribute(tree, ["feature_importances_", "feature_importances"])


def weights_and_bias(svm):
    """(w, b) of a linear SVM, ours or a fitted scikit-learn Pipeline, or None."""
    try:
        w, b = benchmark.linear_weights(svm)
    except (AttributeError, TypeError, ValueError):
        return None
    return (w, b) if np.ndim(w) == 1 else None


def objective_history(svm):
    """The training objective J recorded during fitting (one value per step or epoch), or None."""
    return _first_attribute(svm, ["objective_history_", "history_", "losses_"])
