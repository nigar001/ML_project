"""Tests for our from-scratch DecisionTree (src/decision_tree.py)."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.benchmark import tree_summary
from src.decision_tree import DecisionTree


def _leaves(tree) -> int:
    return tree_summary(tree)["Leaves"]


@pytest.fixture
def data():
    """Continuous features, so no two splits tie and the tree is fully determined."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    y_cls = ((X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.7, 500)) > 0.5).astype(int)
    y_reg = 2 * X[:, 0] + np.sin(3 * X[:, 1]) + rng.normal(0, 0.3, 500)
    return X, y_cls, y_reg, rng.normal(size=(300, 5))


# --- impurity formulas -------------------------------------------------------
def test_impurities_match_hand_calculations():
    gini, entropy = DecisionTree("classification", "gini"), DecisionTree("classification", "entropy")
    mse = DecisionTree("regression", "mse")
    assert gini._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    assert gini._impurity(np.array([0, 0, 0, 1])) == pytest.approx(0.375)
    assert gini._impurity(np.array([1, 1, 1])) == 0.0
    assert entropy._impurity(np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert entropy._impurity(np.array([0, 0, 0, 1])) == pytest.approx(0.8112781244591328)
    assert entropy._impurity(np.array([1, 1, 1])) == 0.0
    assert mse._impurity(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.25)


# --- split search ------------------------------------------------------------
def test_best_split_on_a_tiny_determined_example_matches_sklearn():
    X = np.array([[1.0, 5.0], [2.0, 1.0], [3.0, 4.0], [10.0, 2.0], [11.0, 6.0], [12.0, 3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    ours = DecisionTree("classification", "gini").fit(X, y)
    sk = DecisionTreeClassifier(random_state=0).fit(X, y)
    assert ours.root.feature == int(sk.tree_.feature[0]) == 0
    assert ours.root.threshold == pytest.approx(float(sk.tree_.threshold[0]))   # 6.5
    assert ours._best_split(X, y)[2] == pytest.approx(0.5)                     # parent 0.5, children 0
    assert list(ours.predict(X)) == list(sk.predict(X))


@pytest.mark.parametrize("task,criterion", [("classification", "gini"),
                                            ("classification", "entropy"),
                                            ("regression", "mse")])
@pytest.mark.parametrize("min_leaf", [1, 3])
def test_fast_split_search_equals_the_simple_reference(task, criterion, min_leaf):
    """_best_split (sorted, cumulative sums) must find the same gain as trying every threshold."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        n, p = int(rng.integers(2, 50)), int(rng.integers(1, 5))
        X = rng.integers(0, 6, size=(n, p)).astype(float)                       # many ties
        y = rng.normal(size=n) * 50 + 1000 if task == "regression" else rng.integers(0, 3, n)
        tree = DecisionTree(task, criterion, min_samples_leaf=min_leaf)
        parent, best = tree._impurity(y), -np.inf
        for f in range(p):
            for thr in tree._candidate_thresholds(X[:, f]):
                gain = tree._impurity_decrease(X[:, f], y, thr, parent)
                if gain is not None:
                    best = max(best, gain)
        got = tree._best_split(X, y)
        if best == -np.inf:
            assert got is None
        else:
            assert got[2] == pytest.approx(best, abs=1e-9)


def test_neighbouring_floats_do_not_create_an_empty_child():
    low, high = 1.0, np.nextafter(1.0, 2.0)          # their midpoint rounds to `high`
    X, y = np.array([[low], [low], [high], [high]]), np.array([0, 0, 1, 1])
    assert list(DecisionTree("classification", "gini").fit(X, y).predict(X)) == [0, 0, 1, 1]


# --- agreement with scikit-learn ---------------------------------------------
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classifier_matches_sklearn(data, criterion):
    X, y, _, X_new = data
    # Shallow tree with big nodes: the best split is unique, so the trees must be identical.
    # (Deep trees can differ where two splits have exactly equal gain: see the tie test below.)
    ours = DecisionTree("classification", criterion, max_depth=4, min_samples_leaf=10).fit(X, y)
    sk = DecisionTreeClassifier(criterion=criterion, max_depth=4, min_samples_leaf=10,
                                random_state=42).fit(X, y)
    assert np.array_equal(ours.predict(X_new), sk.predict(X_new))
    assert np.allclose(ours.predict_proba(X_new), sk.predict_proba(X_new))
    assert tree_summary(ours)["Leaves"] == tree_summary(sk)["Leaves"]


def test_regressor_matches_sklearn(data):
    X, _, y, X_new = data
    ours = DecisionTree("regression", "mse", max_depth=6, min_samples_leaf=5).fit(X, y)
    sk = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42).fit(X, y)
    assert np.allclose(ours.predict(X_new), sk.predict(X_new))


def test_unconstrained_tree_is_equivalent_to_sklearn_up_to_ties(data):
    """Deep trees may pick different but EQUALLY good splits (tie-breaking), so the leaf count
    can differ by a leaf or two. What must hold: same root split, same depth, perfect fit."""
    X, y, _, _ = data
    ours = DecisionTree("classification", "gini").fit(X, y)
    sk = DecisionTreeClassifier(random_state=42).fit(X, y)
    assert np.array_equal(ours.predict(X), y)
    assert ours.root.feature == int(sk.tree_.feature[0])
    assert ours.root.threshold == pytest.approx(float(sk.tree_.threshold[0]), abs=1e-6)
    assert tree_summary(ours)["Depth"] == tree_summary(sk)["Depth"]
    assert abs(tree_summary(ours)["Leaves"] - tree_summary(sk)["Leaves"]) <= 2


# --- every stopping rule must change the tree --------------------------------
def test_each_hyperparameter_changes_the_tree(data):
    X, y, _, _ = data
    base = _leaves(DecisionTree("classification", "gini").fit(X, y))
    for limit in ({"max_depth": 3}, {"min_samples_split": 60},
                  {"min_samples_leaf": 25}, {"min_impurity_decrease": 0.01}):
        small = _leaves(DecisionTree("classification", "gini", **limit).fit(X, y))
        assert small < base, limit
    assert tree_summary(DecisionTree("classification", "gini", max_depth=3).fit(X, y))["Depth"] == 3


def test_min_impurity_decrease_is_weighted_like_sklearn(data):
    X, y, _, _ = data
    kw = {"min_impurity_decrease": 0.01, "max_depth": 8}
    ours = DecisionTree("classification", "gini", **kw).fit(X, y)
    sk = DecisionTreeClassifier(random_state=42, **kw).fit(X, y)
    assert _leaves(ours) == tree_summary(sk)["Leaves"]


# --- outputs -----------------------------------------------------------------
def test_predict_proba_are_leaf_frequencies(data):
    X, y, _, X_new = data
    tree = DecisionTree("classification", "gini", max_depth=4).fit(X, y)
    proba = tree.predict_proba(X_new)
    assert proba.shape == (len(X_new), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(tree.predict(X_new), tree.classes_[np.argmax(proba, axis=1)])


def test_feature_importances_sum_to_one_and_find_the_signal(data):
    X, y, y_reg, _ = data
    for model in (DecisionTree("classification", "gini", max_depth=5).fit(X, y),
                  DecisionTree("regression", "mse", max_depth=5).fit(X, y_reg)):
        importances = model.feature_importances_
        assert importances.shape == (X.shape[1],)
        assert importances.sum() == pytest.approx(1.0)
        assert int(np.argmax(importances)) == 0                    # feature 0 drives both targets
    ours = DecisionTree("regression", "mse", max_depth=3, min_samples_leaf=10).fit(X, y_reg)
    sk = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42).fit(X, y_reg)
    assert np.allclose(ours.feature_importances_, sk.feature_importances_, atol=1e-9)


def test_a_tree_that_never_splits_has_zero_importances():
    tree = DecisionTree("classification", "gini").fit(np.arange(6.0).reshape(-1, 1), np.ones(6))
    assert tree.root.is_leaf() and tree.feature_importances_.sum() == 0.0


def test_input_checks():
    with pytest.raises(RuntimeError):
        DecisionTree().predict(np.zeros((1, 2)))
    tree = DecisionTree("regression", "mse").fit(np.zeros((4, 2)), np.arange(4.0))
    with pytest.raises(ValueError):
        tree.predict(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        DecisionTree("classification", "mse")