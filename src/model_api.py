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
    """
    The training objective J recorded during fitting, as (steps, values), or None.

    Accepts either a plain list of J values (one per recorded step) or a list of
    (step, J) pairs, which is what our PegasosSVM stores in history_.
    """
    history = _first_attribute(svm, ["objective_history_", "history_", "losses_"])
    if history is None or len(history) < 2:
        return None
    if history.ndim == 2 and history.shape[1] == 2:                 # (step, J) pairs
        return history[:, 0], history[:, 1]
    if history.ndim == 1:
        return np.arange(1, len(history) + 1, dtype=float), history
    return None
