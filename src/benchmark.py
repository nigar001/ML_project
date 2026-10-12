"""
benchmark.py — fair comparison of OUR models against scikit-learn baselines.

scikit-learn is used here ONLY as the baseline we compare against (Problem 5).
Our decision tree and our SVM live in decision_tree.py and svm.py.

What "fair" means in this file:
  - every model sees the same rows, the same features and the same folds,
  - hyperparameters are matched (see CONFIG and the baseline builders below),
  - nothing is learned from held-out rows: the price-tier threshold and the
    scaling statistics come from the training part of each fold only,
  - every score is reported as mean ± std over K folds, with fit/predict time.

Run from the repo root:
    python -m src.benchmark --synthetic     # stand-in data, to check the machinery
    python -m src.benchmark                 # real data (needs a working data stage)
    python -m src.benchmark --exact-svm     # also run the slow exact SVM solver

A model that is not written yet (its fit raises NotImplementedError) is listed
as "not implemented yet" and skipped, so the benchmark runs at every stage of
the project.

Assumption: the data stage delivers ONE row per listing. If repeat scrapes of a
listing were kept, plain folds would put the same flat in train and test.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep, evaluate, synthetic
from .decision_tree import DecisionTree
from .svm import PegasosSVM

# --- Config -----------------------------------------------------------------
SEED = data_prep.SEED
N_FOLDS = 5

# One place for every matched setting. Placeholders until the final values are
# chosen from validation evidence (Problem 4).
CONFIG = {
    "tree": {
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "min_impurity_decrease": 0.0,
    },
    "tree_criterion": "gini",       # "gini" | "entropy" (Task B)
    "svm": {"lambda_": 1e-4},
    "log_price": True,              # Task A: model log(price) instead of price
}

# Guard for the real data stage: a sound feature table built from PROBE_ROWS
# rows has far fewer columns than MAX_FEATURE_COLUMNS.
PROBE_ROWS = 1000
MAX_FEATURE_COLUMNS = 2000


# --- scikit-learn baselines with matched settings ---------------------------
def sklearn_tree(task: str, criterion: str = "gini", **limits):
    """Library tree with the same stopping rules as ours ("mse" is "squared_error" there)."""
    if task == "regression":
        return DecisionTreeRegressor(criterion="squared_error", random_state=SEED, **limits)
    return DecisionTreeClassifier(criterion=criterion, random_state=SEED, **limits)


def sklearn_sgd_svm(lambda_: float):
    """
    Linear SVM trained by stochastic gradient descent, like Pegasos.
    SGDClassifier minimises  alpha/2 ||w||^2 + mean hinge loss, so alpha = lambda.
    """
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=lambda_, random_state=SEED)
    return make_pipeline(StandardScaler(), model)


def sklearn_linear_svc(lambda_: float, n_train: int):
    """
    Linear SVM solved exactly. LinearSVC minimises  1/2 ||w||^2 + C * sum hinge;
    dividing by C*n gives our objective with lambda = 1 / (C * n).
    (It also penalises the bias slightly, which our objective does not.)

    Optional (--exact-svm): with a small lambda it needs about a minute per fold
    on the real data and may still stop before converging.
    """
    C = 1.0 / (lambda_ * n_train)
    # A high iteration cap: this solver is the "exact answer", so it must converge.
    model = LinearSVC(loss="hinge", C=C, dual=True, max_iter=1_000_000, random_state=SEED)
    return make_pipeline(StandardScaler(), model)


def sklearn_rbf_svc(C: float = 1.0, gamma="scale"):
    """Kernel SVM, the reference for our non-linear extension. Slow: use on a subsample."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma))


def _spec(name: str, make, scale: bool = False, baseline: str | None = None) -> dict:
    """
    One benchmark entry. `make(n_train)` builds a fresh, unfitted model.
    `baseline` is set on OUR models only: the name of the library model to compare with.
    """
    return {"name": name, "make": make, "scale": scale, "baseline": baseline}


def model_specs(task: str, config: dict = CONFIG, exact_svm: bool = False) -> list[dict]:
    """Our models and their baselines for one task, in table order."""
    limits = config["tree"]
    if task == "regression":
        return [
            _spec("Ours: tree", baseline="sklearn: DecisionTreeRegressor",
                  make=lambda n: DecisionTree(
                      task="regression", criterion="mse", random_state=SEED, **limits)),
            _spec("sklearn: DecisionTreeRegressor",
                  make=lambda n: sklearn_tree("regression", **limits)),
        ]

    criterion = config["tree_criterion"]
    lambda_ = config["svm"]["lambda_"]
    specs = [
        _spec("Ours: tree", baseline="sklearn: DecisionTreeClassifier",
              make=lambda n: DecisionTree(
                  task="classification", criterion=criterion, random_state=SEED, **limits)),
        _spec("sklearn: DecisionTreeClassifier",
              make=lambda n: sklearn_tree("classification", criterion, **limits)),
        # Our SVM gets features standardized by the harness; the sklearn SVMs
        # standardize inside their own Pipeline.
        _spec("Ours: SVM (Pegasos)", baseline="sklearn: SGDClassifier", scale=True,
              make=lambda n: PegasosSVM(lambda_=lambda_, random_state=SEED)),
        _spec("sklearn: SGDClassifier", make=lambda n: sklearn_sgd_svm(lambda_)),
    ]
    if exact_svm:
        specs.append(_spec("sklearn: LinearSVC", make=lambda n: sklearn_linear_svc(lambda_, n)))
    return specs


# --- Folds (our own NumPy code) ---------------------------------------------
def kfold_indices(n: int, k: int = N_FOLDS, seed: int = SEED) -> list[np.ndarray]:
    """Shuffle 0..n-1 and cut it into k test folds of (almost) equal size."""
    order = np.random.default_rng(seed).permutation(n)
    return [np.sort(fold) for fold in np.array_split(order, k)]


def stratified_kfold_indices(y, k: int = N_FOLDS, seed: int = SEED) -> list[np.ndarray]:
    """Like kfold_indices, but every fold keeps the same share of each class."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds: list[list[np.ndarray]] = [[] for _ in range(k)]
    for label in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == label))
        for i, part in enumerate(np.array_split(rows, k)):
            folds[i].append(part)
    return [np.sort(np.concatenate(parts)) for parts in folds]


def tier_labels_from_train(price_train, price_test):
    """
    Premium (1) / standard (0) labels for both parts, using the median of the
    TRAINING prices only. Returns (y_train, y_test, threshold).
    """
    y_train, threshold = data_prep.make_tier_label(price_train)
    y_test, _ = data_prep.make_tier_label(price_test, threshold)
    return y_train, y_test, threshold


# --- Fit, predict, time ------------------------------------------------------
def ranking_scores(model, X):
    """Scores for ROC-AUC: probability of premium if the model has one, else the margin."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        return proba[:, 1] if proba.ndim == 2 else proba
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return None


def timed_fit_predict(model, X_train, y_train, X_test, want_scores: bool = False):
    """Return (predictions, scores, fit_seconds, predict_seconds)."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = np.asarray(model.predict(X_test))
    predict_seconds = time.perf_counter() - start

    scores = ranking_scores(model, X_test) if want_scores else None
    return predictions, scores, fit_seconds, predict_seconds


def score_split(spec, X_train, price_train, X_test, price_test, task, log_price=True) -> dict:
    """Fit one model on one train/test split; return its metrics and timings."""
    if spec["scale"]:
        X_train, X_test = data_prep.standardize(X_train, X_test)    # train statistics only

    if task == "classification":
        y_train, y_test, _ = tier_labels_from_train(price_train, price_test)
    elif log_price:
        y_train, y_test = np.log(price_train), np.log(price_test)
    else:
        y_train, y_test = price_train, price_test

    model = spec["make"](len(X_train))
    pred, scores, fit_seconds, predict_seconds = timed_fit_predict(
        model, X_train, y_train, X_test, want_scores=(task == "classification"))

    if task == "classification":
        metrics = evaluate.classification_metrics(y_test, pred, scores)
    elif log_price:
        metrics = {
            "RMSE (log)": evaluate.rmse(y_test, pred),
            "MAE (log)": evaluate.mae(y_test, pred),
            "R2 (log)": evaluate.r2(y_test, pred),
            "MAE (AZN)": evaluate.mae(price_test, np.exp(pred)),   # back on the price scale
        }
    else:
        metrics = evaluate.regression_metrics(y_test, pred)
    metrics["Fit s"] = fit_seconds
    metrics["Predict s"] = predict_seconds
    return metrics


def cross_validate(spec, X, y_price, task, k=N_FOLDS, seed=SEED, log_price=True):
    """
    K-fold cross-validation of one model. Returns {metric: (mean, std)}, or None
    if the model is not implemented yet.
    """
    X = np.asarray(X, dtype=float)
    y_price = np.asarray(y_price, dtype=float)
    if task == "classification":
        # Folds are balanced on a provisional label; the labels the model
        # actually trains on are rebuilt inside each fold from its train median.
        provisional, _ = data_prep.make_tier_label(y_price)
        folds = stratified_kfold_indices(provisional, k, seed)
    else:
        folds = kfold_indices(len(y_price), k, seed)

    per_fold = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        try:
            per_fold.append(score_split(
                spec, X[train_idx], y_price[train_idx], X[test_idx], y_price[test_idx],
                task, log_price))
        except NotImplementedError:
            return None

    return {
        name: (float(np.mean([m[name] for m in per_fold])),
               float(np.std([m[name] for m in per_fold])))
        for name in per_fold[0]
    }


# --- How close are our models to the library's? -----------------------------
def prediction_agreement(pred_ours, pred_baseline, task: str) -> dict:
    """Share of identical tier calls, or the size of the gap between price predictions."""
    pred_ours, pred_baseline = np.asarray(pred_ours), np.asarray(pred_baseline)
    if task == "classification":
        return {"Same prediction %": float(np.mean(pred_ours == pred_baseline) * 100)}
    gap = np.abs(pred_ours - pred_baseline)
    return {"Mean |gap|": float(np.mean(gap)), "Max |gap|": float(np.max(gap))}


def hinge_objective(w, b, X, y_pm1, lambda_: float) -> float:
    """J(w, b) = lambda/2 ||w||^2 + mean(max(0, 1 - y (X w + b))), labels in {-1, +1}."""
    w = np.asarray(w, dtype=float)
    margins = np.asarray(y_pm1) * (np.asarray(X, dtype=float) @ w + b)
    return float(lambda_ / 2.0 * (w @ w) + np.mean(np.maximum(0.0, 1.0 - margins)))


def linear_weights(model):
    """(w, b) of a linear SVM: ours (attributes w, b) or a fitted sklearn Pipeline."""
    if hasattr(model, "steps"):
        linear = model.steps[-1][1]
        return linear.coef_.ravel(), float(linear.intercept_[0])
    return np.asarray(model.w, dtype=float), float(model.b)


def tree_summary(model, feature_names=None) -> dict:
    """Depth, leaf count and the root split of a fitted tree (ours or sklearn's)."""
    if hasattr(model, "tree_"):                                     # scikit-learn
        depth, leaves = int(model.get_depth()), int(model.get_n_leaves())
        feature, threshold = int(model.tree_.feature[0]), float(model.tree_.threshold[0])
    else:                                                           # our Node structure
        def walk(node, level):
            if node.is_leaf():
                return level, 1
            left_depth, left_leaves = walk(node.left, level + 1)
            right_depth, right_leaves = walk(node.right, level + 1)
            return max(left_depth, right_depth), left_leaves + right_leaves

        depth, leaves = walk(model.root, 0)
        feature, threshold = model.root.feature, model.root.threshold

    name = feature_names[feature] if feature_names is not None and feature is not None else feature
    return {"Depth": depth, "Leaves": leaves, "Root feature": name, "Root threshold": threshold}


def compare_pair(ours, baseline, X_train, price_train, X_test, price_test, task,
                 log_price=True, feature_names=None, lambda_=None):
    """
    Fit our model and its baseline on the same split and describe the gap.
    Returns None if our model is not implemented yet.
    """
    if task == "classification":
        y_train, _, _ = tier_labels_from_train(price_train, price_test)
    else:
        y_train = np.log(price_train) if log_price else price_train

    fitted, predictions = {}, {}
    for spec in (ours, baseline):
        X_tr, X_te = (data_prep.standardize(X_train, X_test) if spec["scale"]
                      else (X_train, X_test))
        model = spec["make"](len(X_train))
        try:
            model.fit(X_tr, y_train)
        except NotImplementedError:
            return None
        fitted[spec["name"]], predictions[spec["name"]] = model, np.asarray(model.predict(X_te))

    report = prediction_agreement(predictions[ours["name"]], predictions[baseline["name"]], task)
    for name, model in fitted.items():
        if hasattr(model, "tree_") or hasattr(model, "root"):
            report[name] = tree_summary(model, feature_names)
        elif lambda_ is not None:
            # Both SVMs work on standardized features, so their weights are comparable.
            X_std = data_prep.standardize(X_train)[0]
            w, b = linear_weights(model)
            report[name] = {"Objective J": hinge_objective(w, b, X_std, 2 * y_train - 1, lambda_),
                            "||w||": float(np.linalg.norm(w))}
    return report


def timing_by_size(spec, X, y_price, task, sizes, seed=SEED, log_price=True):
    """Fit and predict time on the first n rows of one seeded shuffle, for each n in sizes."""
    order = np.random.default_rng(seed).permutation(len(X))
    timings = {}
    for n in sizes:
        rows = order[:n]
        try:
            metrics = score_split(spec, X[rows], y_price[rows], X[rows], y_price[rows],
                                  task, log_price)
        except NotImplementedError:
            return None
        timings[f"fit s @{n:,}"] = metrics["Fit s"]
        timings[f"predict s @{n:,}"] = metrics["Predict s"]
    return timings


# --- The whole benchmark -----------------------------------------------------
def run(X, y_price, feature_names=None, config: dict = CONFIG, k: int = N_FOLDS,
        exact_svm: bool = False) -> dict:
    """Run both tasks, print the tables, and return every number in one dict."""
    X = np.asarray(X, dtype=float)
    y_price = np.asarray(y_price, dtype=float)
    log_price = config["log_price"]
    results = {"task_a": {}, "task_b": {}, "skipped": [], "agreement": {}, "timing": {}}

    target = "log price" if log_price else "price"
    titles = {
        "regression": ("task_a", f"Task A: price regression (target: {target})"),
        "classification": ("task_b", "Task B: price tier (premium vs standard)"),
    }
    for task, (key, title) in titles.items():
        specs = model_specs(task, config, exact_svm)
        for spec in specs:
            scores = cross_validate(spec, X, y_price, task, k, log_price=log_price)
            if scores is None:
                results["skipped"].append(f"{spec['name']} ({key})")
            else:
                results[key][spec["name"]] = scores

        print(f"\n== {title}: {k}-fold cross-validation, mean ± std ==")
        evaluate.compare(results[key])

        # Our model against its closest baseline, on one held-out fold.
        test_idx = kfold_indices(len(X), k)[0]
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
        by_name = {spec["name"]: spec for spec in specs}
        for our_spec in [spec for spec in specs if spec["baseline"]]:
            baseline = by_name[our_spec["baseline"]]
            report = compare_pair(
                our_spec, baseline, X[train_idx], y_price[train_idx], X[test_idx],
                y_price[test_idx], task, log_price, feature_names,
                lambda_=config["svm"]["lambda_"])
            if report is not None:
                results["agreement"][f"{our_spec['name']} vs {baseline['name']} ({key})"] = report

        # Time against row count: our models and the baseline each is matched with.
        # (The exact SVM is left out: its C = 1/(lambda*n) changes with n, so its
        # time does not show how the solver scales.)
        sizes = [n for n in (1000, 5000, 20000) if n <= len(X)] or [len(X)]
        matched = {spec["baseline"] for spec in specs if spec["baseline"]}
        for spec in specs:
            if not spec["baseline"] and spec["name"] not in matched:
                continue
            timings = timing_by_size(spec, X, y_price, task, sizes, log_price=log_price)
            if timings is not None:
                results["timing"][f"{spec['name']} ({key})"] = timings

    print("\n== Fit and predict time by number of rows (seconds) ==")
    evaluate.compare(results["timing"])

    print("\n== Agreement between our models and scikit-learn (one held-out fold) ==")
    if results["agreement"]:
        for pair, report in results["agreement"].items():
            print(f"{pair}: {report}")
    else:
        print("Nothing to compare yet.")

    if results["skipped"]:
        print("\nNot implemented yet, skipped: " + "; ".join(results["skipped"]))
    return results


def load_real_data():
    """Features from the real data stage, refusing to continue if it is not ready."""
    df = data_prep.clean(data_prep.load_raw())

    if "estate_rel_url_x" in df.columns and df["estate_rel_url_x"].duplicated().any():
        repeats = int(df["estate_rel_url_x"].duplicated().sum())
        raise SystemExit(
            f"Data stage not ready: {repeats:,} rows are repeat scrapes of a listing already "
            "in the table, so folds would put the same flat in train and test.")

    # Try a small sample first: a feature table that one-hot encodes ids or free
    # text shows up here as thousands of columns, before it can exhaust memory.
    probe_columns = data_prep.make_features(df.head(PROBE_ROWS))[0].shape[1]
    if probe_columns > MAX_FEATURE_COLUMNS:
        raise SystemExit(
            f"Data stage not ready: make_features creates {probe_columns:,} columns from only "
            f"{PROBE_ROWS:,} rows, so text or id columns are being one-hot encoded.")

    X, y_price, feature_names = data_prep.make_features(df)
    if np.isnan(np.asarray(X, dtype=float)).any():
        raise SystemExit("Data stage not ready: the feature table contains missing values.")
    return X, y_price, feature_names


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark our models against scikit-learn.")
    parser.add_argument("--synthetic", action="store_true",
                        help="use seeded stand-in listings instead of the real data")
    parser.add_argument("--exact-svm", action="store_true",
                        help="also run LinearSVC, the exact solver (slow on the real data)")
    args = parser.parse_args(argv)

    if args.synthetic:
        print("SYNTHETIC DATA: checks the benchmark machinery only. Not project results.")
        X, y_price, feature_names = synthetic.make_listings()
    else:
        X, y_price, feature_names = load_real_data()
    run(X, y_price, feature_names, exact_svm=args.exact_svm)


if __name__ == "__main__":
    main()
