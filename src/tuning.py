"""
tuning.py — choosing the final settings on the VALIDATION split (Problem 4).

run_all calls run() with the training and validation rows of its one split.
The test rows never reach this file. run() returns the chosen settings, which
run_all lays over benchmark.CONFIG for every later number.

  SVM:  sweep lambda; keep the value with the best validation F1.
        (Member 3's wider SVM studies — schedules, batch size, random Fourier
        features — are in svm_experiments.py; run_all runs them too.)
  Tree: Gini against entropy and a max_depth sweep (Task B), a max_depth sweep
        (Task A), then min_samples_leaf at the chosen depth. Skipped while the
        team's tree is still a template; the placeholders are kept then.
"""

from __future__ import annotations

import os

import numpy as np

from . import data_prep, evaluate, report_figures
from .decision_tree import DecisionTree
from .svm import PegasosSVM

SEED = data_prep.SEED
SVM_LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SVM_EPOCHS = 20                     # passes over the training rows for every SVM fit
TREE_DEPTHS = [2, 4, 6, 8, 10, 12, 15]
TREE_MIN_LEAF = [1, 5, 20, 50]
TREE_MAX_ROWS = 20_000              # the tree sweeps use a seeded sample of at most this many rows


def _f1(y_true, y_pred) -> float:
    return evaluate.precision_recall_f1(y_true, y_pred)[2]


def svm_study(X_train, y_train, X_val, y_val, fig_dir, svm_class=PegasosSVM) -> dict:
    """lambda sweep of our SVM. Returns the chosen settings and the sweep table."""
    S_train, S_val = data_prep.standardize(X_train, X_val)          # training statistics only
    rows = []
    for lambda_ in SVM_LAMBDAS:
        model = svm_class(lambda_=lambda_, n_epochs=SVM_EPOCHS, random_state=SEED).fit(S_train, y_train)
        norm = float(np.linalg.norm(model.w))
        rows.append({"lambda": lambda_, "train F1": _f1(y_train, model.predict(S_train)),
                     "val F1": _f1(y_val, model.predict(S_val)),
                     "margin": 2.0 / norm if norm > 0 else float("inf")})
    best = max(rows, key=lambda row: row["val F1"])["lambda"]

    lambdas = [row["lambda"] for row in rows]
    report_figures.sweep_plot(
        lambdas, {"training": [r["train F1"] for r in rows], "validation": [r["val F1"] for r in rows]},
        "Our SVM: F1 against lambda", "lambda (log scale)", "F1 (premium tier)",
        os.path.join(fig_dir, "tuning_svm_lambda_f1.png"), log_x=True, mark_x=best)
    report_figures.sweep_plot(
        lambdas, {"margin width 2/||w||": [r["margin"] for r in rows]},
        "Our SVM: margin width against lambda", "lambda (log scale)", "Margin width",
        os.path.join(fig_dir, "tuning_svm_lambda_margin.png"), log_x=True, mark_x=best)
    return {"settings": {"lambda_": best, "n_epochs": SVM_EPOCHS}, "sweep": rows}


def _tree_scores(tree_class, task, criterion, X_train, y_train, X_val, y_val, **limits):
    """(training score, validation score): F1 for Task B, R2 on log price for Task A."""
    model = tree_class(task=task, criterion=criterion, random_state=SEED, **limits).fit(X_train, y_train)
    score = _f1 if task == "classification" else evaluate.r2
    return score(y_train, model.predict(X_train)), score(y_val, model.predict(X_val))


def tree_study(X_train, price_train, X_val, price_val, fig_dir, tree_class=DecisionTree):
    """Depth and leaf-size sweeps of our tree, or None while it is not implemented."""
    keep = np.sort(np.random.default_rng(SEED).permutation(len(X_train))[:TREE_MAX_ROWS])
    X_tr, p_tr = X_train[keep], price_train[keep]
    tier_tr, threshold = data_prep.make_tier_label(price_train)      # threshold: all training rows
    tier_tr, tier_val = tier_tr[keep], data_prep.make_tier_label(price_val, threshold)[0]
    log_tr, log_val = np.log(p_tr), np.log(price_val)
    base = {"min_samples_split": 2, "min_samples_leaf": 5, "min_impurity_decrease": 0.0}

    try:
        curves = {}
        for criterion in ("gini", "entropy"):
            curves[criterion] = [_tree_scores(tree_class, "classification", criterion, X_tr, tier_tr,
                                              X_val, tier_val, max_depth=d, **base)
                                 for d in TREE_DEPTHS]
        curves["mse"] = [_tree_scores(tree_class, "regression", "mse", X_tr, log_tr, X_val, log_val,
                                      max_depth=d, **base) for d in TREE_DEPTHS]
    except NotImplementedError:
        return None

    best_b = {c: max(range(len(TREE_DEPTHS)), key=lambda i: curves[c][i][1]) for c in ("gini", "entropy")}
    criterion = max(best_b, key=lambda c: curves[c][best_b[c]][1])
    # One set of stopping settings serves both tasks: the depth is chosen on Task A,
    # the main task of the tree, and the criterion on Task B.
    depth = TREE_DEPTHS[max(range(len(TREE_DEPTHS)), key=lambda i: curves["mse"][i][1])]
    leaf_scores = [_tree_scores(tree_class, "regression", "mse", X_tr, log_tr, X_val, log_val,
                                max_depth=depth, **dict(base, min_samples_leaf=leaf))
                   for leaf in TREE_MIN_LEAF]
    leaf = TREE_MIN_LEAF[max(range(len(TREE_MIN_LEAF)), key=lambda i: leaf_scores[i][1])]

    report_figures.sweep_plot(
        TREE_DEPTHS, {f"{c}, {part}": [s[i] for s in curves[c]] for c in ("gini", "entropy")
                      for i, part in ((0, "training"), (1, "validation"))},
        "Our tree, Task B: F1 against max_depth", "max_depth", "F1 (premium tier)",
        os.path.join(fig_dir, "tuning_tree_depth_task_b.png"), mark_x=TREE_DEPTHS[best_b[criterion]])
    report_figures.sweep_plot(
        TREE_DEPTHS, {"training": [s[0] for s in curves["mse"]], "validation": [s[1] for s in curves["mse"]]},
        "Our tree, Task A: R2 on log price against max_depth", "max_depth", "R2 (log price)",
        os.path.join(fig_dir, "tuning_tree_depth_task_a.png"), mark_x=depth)
    report_figures.sweep_plot(
        TREE_MIN_LEAF, {"training": [s[0] for s in leaf_scores], "validation": [s[1] for s in leaf_scores]},
        f"Our tree, Task A: R2 against min_samples_leaf (max_depth {depth})", "min_samples_leaf",
        "R2 (log price)", os.path.join(fig_dir, "tuning_tree_min_leaf.png"), mark_x=leaf)

    settings = dict(base, max_depth=depth, min_samples_leaf=leaf)
    return {"settings": settings, "criterion": criterion,
            "depth_sweep": {c: curves[c] for c in curves}, "leaf_sweep": leaf_scores}


def run(X_train, price_train, X_val, price_val, feature_names, fig_dir,
        tree_class=DecisionTree, svm_class=PegasosSVM) -> dict:
    """Choose the final settings on the validation rows; return them in benchmark.CONFIG's shape."""
    X_train, X_val = np.asarray(X_train, float), np.asarray(X_val, float)
    price_train, price_val = np.asarray(price_train, float), np.asarray(price_val, float)
    os.makedirs(fig_dir, exist_ok=True)
    y_train, threshold = data_prep.make_tier_label(price_train)
    y_val, _ = data_prep.make_tier_label(price_val, threshold)

    chosen = {}
    try:
        svm = svm_study(X_train, y_train, X_val, y_val, fig_dir, svm_class)
        chosen["svm"] = svm["settings"]
        print("SVM lambda sweep (validation F1): " + ", ".join(
            f"{r['lambda']:g}: {r['val F1']:.4f}" for r in svm["sweep"]))
    except NotImplementedError:
        print("Our SVM is not implemented yet: keeping the placeholder lambda.")

    tree = tree_study(X_train, price_train, X_val, price_val, fig_dir, tree_class)
    if tree is None:
        print("Our tree is not implemented yet: keeping the placeholder tree settings.")
    else:
        chosen["tree"], chosen["tree_criterion"] = tree["settings"], tree["criterion"]
    return chosen
