"""The tree-vs-scikit-learn comparison runs end to end and writes its two files."""

import json

from src import synthetic, tree_comparison


def test_tree_comparison_writes_results(tmp_path):
    X, price, names = synthetic.make_listings(n=800)
    results = tree_comparison.run(X, price, names, str(tmp_path))

    assert (tmp_path / "tree_comparison.md").exists()
    saved = json.loads((tmp_path / "tree_comparison.json").read_text(encoding="utf-8"))
    assert saved["settings"]["chosen_on"] == "train + validation only"

    for key in ("task_a", "task_b"):
        models = [m for m in results[key] if m != "agreement"]
        assert models[0] == "Ours: tree" and models[1].startswith("sklearn: DecisionTree")
        for m in models:
            assert {"Fit s", "Predict s"} <= set(results[key][m]["metrics"])
    assert {"RMSE (AZN)", "MAE (AZN)", "R2 (AZN)"} <= set(results["task_a"]["Ours: tree"]["metrics"])
    assert {"Accuracy", "Precision", "Recall", "F1", "ROC-AUC"} <= set(
        results["task_b"]["Ours: tree"]["metrics"])
    assert len(results["task_b"]["Ours: tree"]["confusion_matrix"]) == 2
    # the test rows are fitted once and never reach the tuning study
    assert results["data"]["train"] + results["data"]["val"] + results["data"]["test"] == 800