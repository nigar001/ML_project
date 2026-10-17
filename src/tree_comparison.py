"""
tree_comparison.py — OUR decision tree against scikit-learn's, on the project's own split.

Tree only: nothing here touches the SVM. It reuses the pipeline so the numbers are comparable
with results/results.json:
  - the same stratified train / validation / test rows (run_all.split_rows),
  - settings chosen on train + validation only (tuning.tree_study), never on the test rows,
  - both trees get the SAME settings (benchmark.model_specs) and the same features,
  - each final model is fitted on train + validation and scored ONCE on the test rows.

It adds what the main tables do not show: RMSE / MAE / R2 on the actual price (AZN),
the confusion matrices, and the fit time of the final fit.

Run from the repo root:
    python -m src.tree_comparison                # real data -> results/tree_comparison.{json,md}
    python -m src.tree_comparison --synthetic    # made-up listings -> results/synthetic/ (machinery check)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import benchmark, data_prep, evaluate, run_all, synthetic, tuning

REPEATS = 3          # fits per model; the median fit time is reported


def _tree_specs(task: str, config: dict) -> list[dict]:
    """Our tree and its scikit-learn twin (the SVM entries of model_specs are not used)."""
    return [s for s in benchmark.model_specs(task, config) if "tree" in s["name"].lower()]


def _fit_and_score(spec, X_fit, y_fit, X_test, want_scores: bool):
    """Fit REPEATS times (fresh model each time); return the last model, predictions, scores, times."""
    fit_times, predict_times = [], []
    for _ in range(REPEATS):
        model = spec["make"](len(X_fit))
        pred, scores, fit_s, predict_s = benchmark.timed_fit_predict(
            model, X_fit, y_fit, X_test, want_scores=want_scores)
        fit_times.append(fit_s)
        predict_times.append(predict_s)
    return model, pred, scores, float(np.median(fit_times)), float(np.median(predict_times))


def run(X, y_price, feature_names, out_dir: str, fig_dir: str | None = None, label: str = "") -> dict:
    X = np.asarray(X, dtype=float)
    y_price = np.asarray(y_price, dtype=float)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = fig_dir or os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. The pipeline's split; the premium threshold comes from TRAINING prices only.
    train, val, test = run_all.split_rows(y_price)
    fit_rows = np.sort(np.concatenate([train, val]))
    threshold = float(np.median(y_price[train]))
    tier, _ = data_prep.make_tier_label(y_price, threshold)

    # 2. Settings chosen on train + validation (Gini vs entropy, max_depth, min_samples_leaf).
    tuned = tuning.tree_study(X[train], y_price[train], X[val], y_price[val], fig_dir)
    chosen = {"tree": tuned["settings"], "tree_criterion": tuned["criterion"]} if tuned else {}
    config = run_all.merge_config(benchmark.CONFIG, chosen)
    log_price = config["log_price"]

    results = {"data": {"n": len(X), "n_features": X.shape[1], "train": len(train), "val": len(val),
                        "test": len(test), "threshold": threshold},
               "settings": {"tree": config["tree"], "criterion": config["tree_criterion"],
                            "target": "log(price)" if log_price else "price",
                            "chosen_on": "train + validation only" if tuned else "placeholders"},
               "fit_rows": len(fit_rows), "task_a": {}, "task_b": {}}

    for task, key in (("regression", "task_a"), ("classification", "task_b")):
        if task == "classification":
            y_fit, y_test = tier[fit_rows], tier[test]
        elif log_price:
            y_fit, y_test = np.log(y_price[fit_rows]), np.log(y_price[test])
        else:
            y_fit, y_test = y_price[fit_rows], y_price[test]

        predictions = {}
        for spec in _tree_specs(task, config):
            model, pred, scores, fit_s, predict_s = _fit_and_score(
                spec, X[fit_rows], y_fit, X[test], want_scores=(task == "classification"))
            predictions[spec["name"]] = pred
            if task == "classification":
                metrics = evaluate.classification_metrics(y_test, pred, scores)
                matrix = evaluate.confusion_matrix(y_test, pred).tolist()
            else:
                price_pred = np.exp(pred) if log_price else pred
                azn = evaluate.regression_metrics(y_price[test], price_pred)
                metrics = {f"{name} (AZN)": value for name, value in azn.items()}
                if log_price:
                    metrics.update({"RMSE (log)": evaluate.rmse(y_test, pred),
                                    "MAE (log)": evaluate.mae(y_test, pred),
                                    "R2 (log)": evaluate.r2(y_test, pred)})
                matrix = None
            metrics.update({"Fit s": fit_s, "Predict s": predict_s})
            results[key][spec["name"]] = {
                "metrics": metrics, "confusion_matrix": matrix,
                "structure": benchmark.tree_summary(model, feature_names)}

        names = list(predictions)
        results[key]["agreement"] = benchmark.prediction_agreement(
            predictions[names[0]], predictions[names[1]], task)
        print(f"\n== {label}{'Task A: price regression' if key == 'task_a' else 'Task B: price tier'}"
              f", test split ==")
        evaluate.compare({n: results[key][n]["metrics"] for n in names})

    with open(os.path.join(out_dir, "tree_comparison.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False,
                  default=lambda v: v.item() if hasattr(v, "item") else str(v))
    write_markdown(results, os.path.join(out_dir, "tree_comparison.md"), label)
    print(f"\nSaved {out_dir}/tree_comparison.json and {out_dir}/tree_comparison.md")
    return results


def write_markdown(results: dict, path: str, label: str = "") -> None:
    data, settings = results["data"], results["settings"]
    lines = [f"# {label}Decision tree: ours vs scikit-learn", "",
             f"Listings: {data['n']:,} with {data['n_features']} features. "
             f"Train / validation / test: {data['train']:,} / {data['val']:,} / {data['test']:,}. "
             f"Final models are fitted on train + validation ({results['fit_rows']:,} rows) and scored "
             f"once on the test rows. Premium threshold (training median): {data['threshold']:,.0f} AZN.", "",
             f"Settings ({settings['chosen_on']}): `{json.dumps(settings['tree'])}`, "
             f"criterion `{settings['criterion']}` (Task B), regression target {settings['target']}. "
             "Both trees use the same settings.", ""]

    for key, title in (("task_a", "Task A: price regression"), ("task_b", "Task B: price tier")):
        block = results[key]
        names = [n for n in block if n != "agreement"]
        ours, theirs = names[0], names[1]
        metric_names = list(block[ours]["metrics"])
        lines += [f"## {title}, test split", "",
                  "| Model | " + " | ".join(metric_names) + " |", "|---" * (len(metric_names) + 1) + "|"]
        for name in names:
            cells = [evaluate._format_value(block[name]["metrics"][m]) for m in metric_names]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        diff = [evaluate._format_value(block[ours]["metrics"][m] - block[theirs]["metrics"][m])
                for m in metric_names]
        lines += ["| *ours − sklearn* | " + " | ".join(diff) + " |", ""]

        if key == "task_b":
            lines += ["Confusion matrices, rows = actual, columns = predicted ([[TN, FP], [FN, TP]], "
                      "premium = positive):", ""]
            for name in names:
                (tn, fp), (fn, tp) = block[name]["confusion_matrix"]
                lines.append(f"- {name}: TN {tn:,}, FP {fp:,}, FN {fn:,}, TP {tp:,}")
            lines.append("")

        lines += ["Tree structure (depth, leaves, root split):", ""]
        for name in names:
            s = block[name]["structure"]
            lines.append(f"- {name}: depth {s['Depth']}, {s['Leaves']} leaves, root feature "
                         f"`{s['Root feature']}` at {s['Root threshold']:.4g}")
        lines += ["", "Agreement of test predictions: "
                  + ", ".join(f"{k} = {v:,.4f}" for k, v in block["agreement"].items()), ""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Our decision tree vs scikit-learn's.")
    parser.add_argument("--synthetic", action="store_true",
                        help="use seeded made-up listings (checks the machinery; not project results)")
    args = parser.parse_args(argv)

    if args.synthetic:
        print("SYNTHETIC DATA: checks the machinery only. Not project results.")
        X, y_price, names = synthetic.make_listings()
        run(X, y_price, names, os.path.join(run_all.OUT_DIR, "synthetic"), label="SYNTHETIC DATA. ")
    else:
        X, y_price, names = benchmark.load_real_data()
        run(X, y_price, names, run_all.OUT_DIR, fig_dir=os.path.join(run_all.OUT_DIR, "tree_figures"))


if __name__ == "__main__":
    main()