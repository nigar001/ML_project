"""
test_run_all.py — checks for the one-command pipeline in src/run_all.py.

Runs on small seeded synthetic listings in a temporary folder, so it needs
neither the real dataset nor the team's tree and SVM.

Run from the repo root:
    python -m pytest tests/test_run_all.py -v
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from src import benchmark, run_all, synthetic
from tests.standins import StandInSVM, StandInTree

STANDINS = (StandInTree, StandInSVM)


@pytest.fixture(scope="module")
def listings():
    X, price, names = synthetic.make_listings(n=600, seed=7)
    rows = pd.DataFrame({"category": np.where(X[:, 0] > 90, "big", "small"),
                         "district": np.where(X[:, 4] > 40.377, "north", "south")})
    return X, price, names, rows


def _run(listings, folder, **options):
    X, price, names, rows = listings
    return run_all.run_pipeline(X, price, names, rows, out_dir=str(folder),
                                fig_dir=str(folder / "figures"), k=3, **options)


@pytest.fixture(scope="module")
def standin_run(listings, tmp_path_factory):
    folder = tmp_path_factory.mktemp("standins")
    return _run(listings, folder, model_classes=STANDINS), folder


# --- Helpers ----------------------------------------------------------------
def test_split_is_complete_disjoint_and_seeded(listings):
    price = listings[1]
    train, val, test = run_all.split_rows(price)
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(len(price)))
    assert len(val) == len(test) == int(len(price) * 0.15)
    again = run_all.split_rows(price)
    assert all(np.array_equal(a, b) for a, b in zip((train, val, test), again))


def test_merge_config_overrides_only_what_is_given():
    merged = run_all.merge_config(benchmark.CONFIG, {"tree": {"max_depth": 3}, "log_price": False})
    assert merged["tree"]["max_depth"] == 3 and merged["log_price"] is False
    assert merged["tree"]["min_samples_leaf"] == benchmark.CONFIG["tree"]["min_samples_leaf"]
    assert benchmark.CONFIG["tree"]["max_depth"] != 3              # the defaults are untouched


# --- The pipeline with the team's templates ----------------------------------
def test_pipeline_runs_while_team_models_are_templates(listings, tmp_path, capsys):
    results = _run(listings, tmp_path)
    printed = capsys.readouterr().out

    assert len(results["skipped"]) == 3                             # tree x2 tasks, SVM
    assert list(results["test"]["task_a"]) == ["sklearn: DecisionTreeRegressor"]
    assert set(results["test"]["task_b"]) == {"sklearn: DecisionTreeClassifier",
                                              "sklearn: SGDClassifier"}
    assert "Not implemented yet" in printed and "test split, used once" in printed
    assert (tmp_path / "results.json").exists() and (tmp_path / "summary.md").exists()
    # A model that is not written yet is visible in the report's table, not silently absent.
    assert "Ours: tree & \\multicolumn{6}{c}{not implemented yet}" in (
        tmp_path / "tables" / "results_task_b.tex").read_text()


# --- The pipeline once our models exist (stand-ins) --------------------------
def test_every_table_and_figure_is_produced(standin_run):
    results, folder = standin_run
    assert results["skipped"] == []
    assert set(results["test"]["task_a"]["Ours: tree"]) == {
        "RMSE (log)", "MAE (log)", "R2 (log)", "MAE (AZN)"}
    assert set(results["test"]["task_b"]["Ours: SVM (Pegasos)"]) == {
        "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"}

    svm = results["diagnostics"]["svm"]["Ours: SVM (Pegasos)"]
    assert svm["Margin 2/||w||"] == pytest.approx(2 / svm["||w||"])
    assert 0 < svm["Support vectors"] <= svm["Training points"]
    assert np.sum(results["diagnostics"]["confusion_matrix"]["Ours: tree"]) == results["data"]["test"]
    assert len(results["worst_misses"]["Ours: tree"]) == 10
    for key in ("task_a", "task_b"):
        assert set(results["errors"][key]) == {"price_band", "category", "district_size"}

    expected = ["roc_curves", "confusion_ours_tree", "confusion_ours_svm_pegasos",
                "importances_ours_tree_task_a", "importances_ours_tree_task_b",
                "svm_weights_ours_svm_pegasos", "objective_ours_svm_pegasos",
                "predicted_vs_actual_ours_tree", "error_by_price_band_task_a",
                "error_by_category_task_b", "error_by_district_size_task_a"]
    for name in expected:
        assert (folder / "figures" / f"{name}.png").stat().st_size > 0, name
    assert all(os.path.exists(path) for path in results["figures"])

    tables = {os.path.basename(path) for path in results["tables"]}
    assert {"numbers.tex", "data_summary.tex", "settings.tex", "results_task_a.tex",
            "results_task_b.tex", "errors_task_a.tex", "errors_task_b.tex", "timing.tex",
            "svm_margin.tex", "agreement.tex", "worst_misses.tex"} == tables
    assert "\\newcommand{\\NumListings}{600}" in (folder / "tables" / "numbers.tex").read_text()
    assert "Ours: tree" in (folder / "tables" / "results_task_b.tex").read_text()

    saved = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    assert saved["test"]["task_b"]["Ours: tree"]["F1"] == results["test"]["task_b"]["Ours: tree"]["F1"]
    assert "Task B: price tier, test split" in (folder / "summary.md").read_text(encoding="utf-8")


def test_two_runs_give_identical_numbers(standin_run, listings, tmp_path):
    first, _ = standin_run
    second = _run(listings, tmp_path, model_classes=STANDINS)
    for part in ("test", "diagnostics", "errors", "worst_misses", "data"):
        assert first[part] == second[part], part
    for key in ("task_a", "task_b"):                                # timings differ, scores do not
        for model, scores in first["cv"][key].items():
            for metric, value in scores.items():
                if not metric.endswith(" s"):
                    assert value == second["cv"][key][model][metric]


def test_threshold_is_the_training_median(standin_run, listings):
    results, _ = standin_run
    price = listings[1]
    train, _, _ = run_all.split_rows(price)
    assert results["data"]["threshold"] == float(np.median(price[train]))


# --- The tuning slot ----------------------------------------------------------
def test_tuning_sees_only_train_and_validation_and_its_settings_are_used(listings, tmp_path):
    seen = {}

    def fake_tuning(X_train, price_train, X_val, price_val, feature_names, fig_dir):
        seen.update(train=len(X_train), val=len(X_val), names=feature_names)
        return {"tree": {"max_depth": 2}, "svm": {"lambda_": 0.01}}

    results = _run(listings, tmp_path, model_classes=STANDINS, tuning_run=fake_tuning)
    assert (seen["train"], seen["val"]) == (results["data"]["train"], results["data"]["val"])
    assert seen["train"] + seen["val"] + results["data"]["test"] == results["data"]["n"]
    assert results["tuned"] and results["config"]["tree"]["max_depth"] == 2
    assert results["config"]["svm"]["lambda_"] == 0.01
    assert results["cv"]["agreement"][
        "Ours: tree vs sklearn: DecisionTreeClassifier (task_b)"]["Ours: tree"]["Depth"] <= 2


# --- Optional model outputs ---------------------------------------------------
def test_missing_optional_outputs_skip_their_figures(listings, tmp_path):
    class BareTree(StandInTree):
        def fit(self, X, y):
            super().fit(X, y)
            del self.feature_importances_
            return self

    class BareSVM(StandInSVM):
        def fit(self, X, y):
            super().fit(X, y)
            del self.objective_history_
            return self

    results = _run(listings, tmp_path, model_classes=(BareTree, BareSVM))
    names = [os.path.basename(path) for path in results["figures"]]
    assert "importances_ours_tree_task_b.png" not in names
    assert "objective_ours_svm_pegasos.png" not in names
    assert "svm_weights_ours_svm_pegasos.png" in names               # w and b are still there
    assert "Ours: tree" in results["test"]["task_b"]                 # and the run completed
