"""
standins.py — STAND-IN models for developing and testing the pipeline.

These are NOT the team's models. They have the constructor, methods and
attributes of the templates in src/decision_tree.py and src/svm.py, but borrow
scikit-learn's maths, so the "our models" path of the benchmark and the
pipeline can be built and tested before the real tree and SVM exist.

Nothing produced with them may be reported as a project result. They live in
tests/ for that reason, and `python -m src.run_all --standins` writes to its
own folder with a banner.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from src import benchmark


class StandInTree:
    """Has our tree's constructor and methods, but borrows the library's maths."""

    def __init__(self, task, criterion, random_state, **limits):
        self._model = benchmark.sklearn_tree(task, criterion, **limits)

    def fit(self, X, y):
        self._model.fit(X, y)
        self.tree_ = self._model.tree_
        self.get_depth, self.get_n_leaves = self._model.get_depth, self._model.get_n_leaves
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class StandInSVM:
    """Has our SVM's constructor and attributes (w, b), fitted by the library's SGD."""

    def __init__(self, lambda_, random_state, **training_settings):
        # Training settings meant for our PegasosSVM (e.g. n_epochs) do not apply here.
        self.lambda_, self.random_state = lambda_, random_state
        self._model = SGDClassifier(loss="hinge", alpha=lambda_, random_state=random_state)

    def fit(self, X, y):
        self._model.fit(X, y)
        self.w, self.b = self._model.coef_.ravel(), float(self._model.intercept_[0])

        # The objective after each of a few passes over the data, so the
        # convergence figure has something to draw.
        tracker = SGDClassifier(loss="hinge", alpha=self.lambda_, random_state=self.random_state)
        self.objective_history_ = []
        for _ in range(5):
            tracker.partial_fit(X, y, classes=[0, 1])
            self.objective_history_.append(benchmark.hinge_objective(
                tracker.coef_.ravel(), float(tracker.intercept_[0]), X, 2 * np.asarray(y) - 1,
                self.lambda_))
        return self

    def decision_function(self, X):
        return self._model.decision_function(X)

    def predict(self, X):
        return self._model.predict(X)
