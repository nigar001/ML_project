"""
test_benchmark.py — checks for the benchmark machinery in src/benchmark.py.

These tests run on small seeded synthetic listings (src/synthetic.py), so they
do not need the real dataset or the team's tree and SVM to exist.

Run from the repo root:
    python -m pytest tests/test_benchmark.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src import benchmark, data_prep, synthetic
from src.decision_tree import Node

CLASSIFICATION_METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
TIMINGS = ["Fit s", "Predict s"]


@pytest.fixture(scope="module")
def listings():
    return synthetic.make_listings(n=600, seed=7)


def _sklearn_tree_spec(task):
    """The library tree, taken from the real model list."""
    return next(s for s in benchmark.model_specs(task) if "DecisionTree" in s["name"])


class _NotWrittenYet:
    """Stands in for a team model whose fit is still a template."""

    def fit(self, X, y):
        raise NotImplementedError("TODO")


# --- Synthetic data ---------------------------------------------------------
def test_synthetic_listings_are_seeded_and_clean():
    X1, price1, names = synthetic.make_listings(n=300, seed=1)
    X2, price2, _ = synthetic.make_listings(n=300, seed=1)
    X3, _, _ = synthetic.make_listings(n=300, seed=2)

    assert X1.shape == (300, len(names))
    assert np.array_equal(X1, X2) and np.array_equal(price1, price2)
    assert not np.array_equal(X1, X3)
    assert not np.isnan(X1).any() and (price1 > 0).all()


# --- Folds ------------------------------------------------------------------
def test_kfold_puts_every_row_in_exactly_one_fold():
    folds = benchmark.kfold_indices(103, k=5, seed=0)
    assert len(folds) == 5
    assert sorted(np.concatenate(folds).tolist()) == list(range(103))
    assert max(len(f) for f in folds) - min(len(f) for f in folds) <= 1


def test_stratified_kfold_keeps_the_class_share():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.3).astype(int)
    folds = benchmark.stratified_kfold_indices(y, k=5, seed=0)

    assert sorted(np.concatenate(folds).tolist()) == list(range(1000))
    for fold in folds:
        assert abs(y[fold].mean() - y.mean()) < 0.01


def test_folds_depend_only_on_the_seed():
    a = benchmark.kfold_indices(50, k=5, seed=3)
    b = benchmark.kfold_indices(50, k=5, seed=3)
    c = benchmark.kfold_indices(50, k=5, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


# --- No information from held-out rows --------------------------------------
def test_tier_threshold_comes_from_training_prices_only():
    price_train = np.array([1.0, 2.0, 3.0, 4.0, 100.0])        # median = 3
    y_train, y_test, threshold = benchmark.tier_labels_from_train(
        price_train, np.array([0.0, 3.0, 1000.0]))
    assert threshold == 3.0
    assert y_train.tolist() == [0, 0, 0, 1, 1]
    assert y_test.tolist() == [0, 0, 1]                        # 3.0 is not above 3.0

    # Very different test prices must not move the threshold.
    _, _, threshold_again = benchmark.tier_labels_from_train(
        price_train, np.array([5000.0, 6000.0, 7000.0]))
    assert threshold_again == 3.0


# --- Matched settings -------------------------------------------------------
def test_sklearn_baselines_use_matched_settings():
    limits = {"max_depth": 4, "min_samples_split": 6, "min_samples_leaf": 3,
              "min_impurity_decrease": 0.01}

    regressor = benchmark.sklearn_tree("regression", **limits)
    assert regressor.criterion == "squared_error"              # our "mse"
    classifier = benchmark.sklearn_tree("classification", "entropy", **limits)
    assert classifier.criterion == "entropy"
    for tree in (regressor, classifier):
        assert {name: getattr(tree, name) for name in limits} == limits

    lambda_, n_train = 1e-3, 800
    sgd = benchmark.sklearn_sgd_svm(lambda_)
    assert isinstance(sgd.steps[0][1], StandardScaler)         # scaling inside the Pipeline
    assert sgd.steps[-1][1].loss == "hinge" and sgd.steps[-1][1].alpha == lambda_

    svc = benchmark.sklearn_linear_svc(lambda_, n_train)
    assert isinstance(svc.steps[0][1], StandardScaler)
    assert svc.steps[-1][1].C == pytest.approx(1 / (lambda_ * n_train))


def test_every_team_model_names_a_baseline_that_exists():
    for task in ("regression", "classification"):
        specs = benchmark.model_specs(task)
        names = {s["name"] for s in specs}
        ours = [s for s in specs if s["baseline"]]
        assert ours, "each task compares at least one of our models"
        assert all(s["baseline"] in names for s in ours)


# --- SVM objective ----------------------------------------------------------
def test_hinge_objective_by_hand():
    X = np.array([[2.0, 0.0], [0.5, 0.0], [-1.0, 0.0]])
    y = np.array([1, 1, -1])
    # w = (1, 0), b = 0 -> margins y(w.x) = 2, 0.5, 1 -> hinge = 0, 0.5, 0 -> mean 1/6
    # regulariser = 0.1 / 2 * ||w||^2 = 0.05
    assert benchmark.hinge_objective([1.0, 0.0], 0.0, X, y, lambda_=0.1) == pytest.approx(
        0.05 + 1 / 6)


def test_exact_solver_reaches_a_lower_objective_than_sgd(listings):
    X, price, _ = listings
    y, _ = data_prep.make_tier_label(price)
    X_std = data_prep.standardize(X)[0]
    lambda_ = 1e-2

    objective = {}
    for name, model in [("exact", benchmark.sklearn_linear_svc(lambda_, len(X))),
                        ("sgd", benchmark.sklearn_sgd_svm(lambda_))]:
        w, b = benchmark.linear_weights(model.fit(X, y))
        objective[name] = benchmark.hinge_objective(w, b, X_std, 2 * y - 1, lambda_)

    assert objective["exact"] < 1.0                            # better than w = 0
    assert objective["exact"] <= objective["sgd"] * 1.01


# --- Cross-validation -------------------------------------------------------
def test_cross_validate_classification_returns_mean_and_std(listings):
    X, price, _ = listings
    scores = benchmark.cross_validate(_sklearn_tree_spec("classification"), X, price,
                                      "classification", k=3)
    assert list(scores) == CLASSIFICATION_METRICS + TIMINGS
    for mean, std in scores.values():
        assert std >= 0
    assert scores["ROC-AUC"][0] > 0.75                         # the data has real signal


def test_cross_validate_regression_reports_log_and_azn(listings):
    X, price, _ = listings
    spec = _sklearn_tree_spec("regression")

    on_log = benchmark.cross_validate(spec, X, price, "regression", k=3, log_price=True)
    assert list(on_log) == ["RMSE (log)", "MAE (log)", "R2 (log)", "MAE (AZN)"] + TIMINGS
    assert on_log["MAE (AZN)"][0] > 1000 > on_log["MAE (log)"][0]

    on_price = benchmark.cross_validate(spec, X, price, "regression", k=3, log_price=False)
    assert list(on_price) == ["RMSE", "MAE", "R2"] + TIMINGS


def test_cross_validate_is_reproducible(listings):
    X, price, _ = listings
    spec = _sklearn_tree_spec("classification")
    a = benchmark.cross_validate(spec, X, price, "classification", k=3)
    b = benchmark.cross_validate(spec, X, price, "classification", k=3)
    assert all(a[name] == b[name] for name in CLASSIFICATION_METRICS)


def test_model_that_is_not_written_yet_is_skipped(listings):
    X, price, _ = listings
    spec = {"name": "Ours: later", "make": lambda n: _NotWrittenYet(), "scale": False,
            "baseline": None}
    assert benchmark.cross_validate(spec, X, price, "classification", k=3) is None
    assert benchmark.timing_by_size(spec, X, price, "classification", sizes=[100]) is None


# --- Our model against the library's ----------------------------------------
def test_identical_models_agree_completely(listings):
    X, price, names = listings
    for task, expected in [("classification", {"Same prediction %": 100.0}),
                           ("regression", {"Mean |gap|": 0.0, "Max |gap|": 0.0})]:
        spec = _sklearn_tree_spec(task)
        twin = dict(spec, name="twin")
        report = benchmark.compare_pair(twin, spec, X[:450], price[:450], X[450:], price[450:],
                                        task, feature_names=names)
        for key, value in expected.items():
            assert report[key] == value
        assert report["twin"] == report[spec["name"]]          # same depth, leaves, root split
        assert report["twin"]["Root feature"] in names


def test_pair_is_skipped_when_our_model_is_not_written(listings):
    X, price, _ = listings
    ours = {"name": "Ours: later", "make": lambda n: _NotWrittenYet(), "scale": False,
            "baseline": "x"}
    report = benchmark.compare_pair(ours, _sklearn_tree_spec("classification"),
                                    X[:450], price[:450], X[450:], price[450:], "classification")
    assert report is None


def test_tree_summary_reads_our_node_structure():
    # A one-question tree built from the team's Node class: "area_m2 <= 80.5?"
    root, left, right = Node(), Node(), Node()
    root.feature, root.threshold, root.left, root.right = 0, 80.5, left, right
    left.value, right.value = 0, 1

    class OurTree:
        pass

    tree = OurTree()
    tree.root = root
    assert benchmark.tree_summary(tree, ["area_m2", "rooms"]) == {
        "Depth": 1, "Leaves": 2, "Root feature": "area_m2", "Root threshold": 80.5}


def test_timing_by_size_reports_each_size(listings):
    X, price, _ = listings
    timings = benchmark.timing_by_size(_sklearn_tree_spec("classification"), X, price,
                                       "classification", sizes=[100, 500])
    assert list(timings) == ["fit s @100", "predict s @100", "fit s @500", "predict s @500"]
    assert all(seconds >= 0 for seconds in timings.values())


# --- The whole run ----------------------------------------------------------
def test_run_reports_or_skips_every_model(listings, capsys):
    X, price, names = listings
    results = benchmark.run(X, price, names, k=3)
    printed = capsys.readouterr().out

    for task, key in [("regression", "task_a"), ("classification", "task_b")]:
        for spec in benchmark.model_specs(task):
            reported = spec["name"] in results[key]
            skipped = f"{spec['name']} ({key})" in results["skipped"]
            assert reported != skipped, f"{spec['name']} must be reported or skipped, not both"
            if not spec["baseline"]:
                assert reported, "scikit-learn baselines always run"
    assert "Task A" in printed and "Task B" in printed and "±" in printed


class _StandInTree:
    """Has our tree's constructor and methods, but borrows the library's maths."""

    def __init__(self, task, criterion, random_state, **limits):
        self._model = benchmark.sklearn_tree(task, criterion, **limits)

    def fit(self, X, y):
        self._model.fit(X, y)
        self.tree_ = self._model.tree_
        self.get_depth, self.get_n_leaves = self._model.get_depth, self._model.get_n_leaves
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class _StandInSVM:
    """Has our SVM's constructor and attributes (w, b), fitted by the library's SGD."""

    def __init__(self, lambda_, random_state):
        self._model = benchmark.SGDClassifier(loss="hinge", alpha=lambda_, random_state=random_state)

    def fit(self, X, y):
        self._model.fit(X, y)
        self.w, self.b = self._model.coef_.ravel(), float(self._model.intercept_[0])
        return self

    def decision_function(self, X):
        return self._model.decision_function(X)

    def predict(self, X):
        return self._model.predict(X)


def test_run_compares_team_models_once_they_exist(listings, monkeypatch, capsys):
    # Stand-ins take the place of the team's classes, so this path is tested
    # before the real tree and SVM are written.
    monkeypatch.setattr(benchmark, "DecisionTree", _StandInTree)
    monkeypatch.setattr(benchmark, "PegasosSVM", _StandInSVM)
    X, price, names = listings
    results = benchmark.run(X, price, names, k=3)
    capsys.readouterr()

    assert results["skipped"] == []
    assert set(results["task_b"]) == {s["name"] for s in benchmark.model_specs("classification")}

    agreement = results["agreement"]
    assert len(agreement) == 3                                 # tree x2 tasks, SVM x1
    tree_pair = agreement["Ours: tree vs sklearn: DecisionTreeClassifier (task_b)"]
    assert tree_pair["Same prediction %"] == 100.0
    assert tree_pair["Ours: tree"] == tree_pair["sklearn: DecisionTreeClassifier"]

    svm_pair = agreement["Ours: SVM (Pegasos) vs sklearn: SGDClassifier (task_b)"]
    assert svm_pair["Same prediction %"] > 99.0
    ours_j = svm_pair["Ours: SVM (Pegasos)"]["Objective J"]
    theirs_j = svm_pair["sklearn: SGDClassifier"]["Objective J"]
    assert ours_j == pytest.approx(theirs_j, rel=1e-6)


# --- Guard on the real data stage -------------------------------------------
def test_real_data_is_refused_while_the_data_stage_is_not_ready(monkeypatch):
    def fake_clean(_):
        return pd.DataFrame({
            "price": [100000.0, 120000.0, 90000.0],
            "estate_rel_url_x": ["/items/1", "/items/1", "/items/2"],   # a repeat scrape
        })

    monkeypatch.setattr(data_prep, "load_raw", lambda: None)
    monkeypatch.setattr(data_prep, "clean", fake_clean)
    with pytest.raises(SystemExit, match="repeat scrapes"):
        benchmark.load_real_data()

    # A feature table that one-hot encodes an id column: one column per row.
    def fake_clean_ids(_):
        n = benchmark.MAX_FEATURE_COLUMNS + 500
        return pd.DataFrame({"price": np.arange(n, dtype=float) + 1,
                             "id_x": [f"id-{i}" for i in range(n)]})

    def one_hot_everything(df):
        X = pd.get_dummies(df.drop(columns=["price"]), dtype=float)
        return X.values, df["price"].values, list(X.columns)

    monkeypatch.setattr(benchmark, "PROBE_ROWS", benchmark.MAX_FEATURE_COLUMNS + 100)
    monkeypatch.setattr(data_prep, "clean", fake_clean_ids)
    monkeypatch.setattr(data_prep, "make_features", one_hot_everything)
    with pytest.raises(SystemExit, match="columns"):
        benchmark.load_real_data()

    # A feature table with a hole in it.
    def fake_clean_ok(_):
        return pd.DataFrame({"price": [100000.0, 120000.0], "area": [50.0, np.nan]})

    def numeric_only(df):
        return df[["area"]].values, df["price"].values, ["area"]

    monkeypatch.setattr(data_prep, "clean", fake_clean_ok)
    monkeypatch.setattr(data_prep, "make_features", numeric_only)
    with pytest.raises(SystemExit, match="missing values"):
        benchmark.load_real_data()
