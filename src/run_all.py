"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all                # real data, real models
    python -m src.run_all --synthetic    # made-up listings: checks the machinery only

Stages:
  1. load + clean the data, build features, make ONE stratified train /
     validation / test split,
  2. draw the EDA figures,
  3. tuning slot: final settings chosen on train + validation (src/tuning.py),
  4. cross-validated benchmark of our models against scikit-learn,
  5. final evaluation on the untouched test split, once,
  6. diagnostics: confusion matrix, ROC, feature importances, SVM weights,
     margin, support vectors,
  7. error analysis: where do the models fail?
  8. save every number (results.json, summary.md), every figure (PNG) and the
     report's tables (LaTeX),
  9. (real data only) Member 3's wider SVM studies in svm_experiments.py.

Everything is seeded from data_prep.SEED. Nothing from the test split sets the
tier threshold, the scaling statistics or any setting. A model that is not
written yet is listed as "not implemented yet" and skipped.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re

import matplotlib

matplotlib.use("Agg")       # draw to files; no screen needed. Must come before pyplot is imported.

import numpy as np
import pandas as pd

from . import (benchmark, data_prep, error_analysis, evaluate, model_api, report_figures,
               report_tables, synthetic)

OUT_DIR = "results"
FIG_DIR = "report/figures"
TABLE_DIR = "report/tables"       # LaTeX tables that report/report.tex pulls in
TASKS = {"regression": "task_a", "classification": "task_b"}


# --- Small helpers ----------------------------------------------------------
def split_rows(y_price, seed: int = data_prep.SEED):
    """
    Row numbers of the train / validation / test parts (70 / 15 / 15).
    The split is stratified so each part has the same share of expensive
    listings. Splitting row NUMBERS (not X itself) keeps each listing's
    category and district attached to it for the error analysis.
    """
    balance_on, _ = data_prep.make_tier_label(y_price)      # only balances the split
    rows = np.arange(len(y_price))
    train, _, val, _, test, _ = data_prep.train_val_test_split(
        rows, y_price, seed=seed, stratify_labels=balance_on)
    return np.sort(train), np.sort(val), np.sort(test)


def merge_config(base: dict, chosen: dict | None) -> dict:
    """Settings from the tuning study laid over the defaults (tree and SVM parts merged)."""
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in (chosen or {}).items():
        if isinstance(value, dict):
            merged[key] = {**merged.get(key, {}), **value}
        else:
            merged[key] = value
    return merged


def load_tuning():
    """The tuning study's run function if src/tuning.py exists, else None."""
    if importlib.util.find_spec("src.tuning") is None:
        return None
    return importlib.import_module("src.tuning").run


def slug(name: str) -> str:
    """'Ours: SVM (Pegasos)' -> 'ours_svm_pegasos', for file names."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fit_for_test(spec: dict, task: str, X_fit, y_fit, X_test):
    """
    Fit one model on the train + validation rows and predict the test rows.
    Returns None if the model is not implemented yet.
    """
    if spec["scale"]:
        X_fit, X_test = data_prep.standardize(X_fit, X_test)        # fit-row statistics only
    model = spec["make"](len(X_fit))
    try:
        model.fit(X_fit, y_fit)
    except NotImplementedError:
        return None
    scores = benchmark.ranking_scores(model, X_test) if task == "classification" else None
    return {"model": model, "pred": np.asarray(model.predict(X_test)), "scores": scores,
            "X_fit": X_fit}


def svm_diagnostics(model, X_fit, y_fit) -> dict | None:
    """||w||, the margin width 2/||w|| and the number of support vectors, or None."""
    weights = model_api.weights_and_bias(model)
    if weights is None or not hasattr(model, "decision_function"):
        return None
    norm = float(np.linalg.norm(weights[0]))
    # Support vectors: training points on or inside the margin, y (w.x + b) <= 1.
    margins = (2 * np.asarray(y_fit) - 1) * np.asarray(model.decision_function(X_fit))
    return {"||w||": norm, "Margin 2/||w||": 2.0 / norm if norm > 0 else float("inf"),
            "Support vectors": int(np.sum(margins <= 1)), "Training points": int(len(margins)),
            "Bias b": float(weights[1])}


def error_tables(error_by_model: dict, price_test, price_train, rows_test, rows_train) -> dict:
    """{table name: DataFrame with 'Listings' and one mean-error column per model}."""
    builders = {"price_band": lambda err: error_analysis.by_price_band(
        price_test, err, price_train)}
    if rows_test is not None and "category" in rows_test:
        builders["category"] = lambda err: error_analysis.by_group(rows_test["category"], err)
    if rows_test is not None and "district" in rows_test:
        builders["district_size"] = lambda err: error_analysis.by_district_size(
            rows_test["district"], err, rows_train["district"])

    tables = {}
    for table_name, build in builders.items():
        combined = None
        for model_name, err in error_by_model.items():
            table = build(err)
            if combined is None:
                combined = table[["Listings"]].copy()
            combined[model_name] = table["Mean error"]
        if combined is not None:
            tables[table_name] = combined
    return tables


def _markdown_table(name_to_metrics: dict, first_column: str = "Model") -> list[str]:
    columns: list[str] = []
    for metrics in name_to_metrics.values():
        columns += [key for key in metrics if key not in columns]
    lines = [f"| {first_column} | " + " | ".join(columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    for name, metrics in name_to_metrics.items():
        cells = [evaluate._format_value(metrics.get(column)) for column in columns]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [""]


def write_summary(results: dict, path: str, label: str = "") -> None:
    """The headline tables as Markdown, for reading and for the report."""
    data = results["data"]
    lines = [f"# {label}Results summary", "",
             f"Listings: {data['n']:,} with {data['n_features']} features. "
             f"Train / validation / test: {data['train']:,} / {data['val']:,} / {data['test']:,}. "
             f"Premium threshold (median of training prices): {data['threshold']:,.0f} AZN.", "",
             f"Settings: `{json.dumps(results['config'])}` "
             f"({'from the tuning study' if results['tuned'] else 'placeholders, no tuning study yet'}).",
             ""]
    titles = {"task_a": "Task A: price regression", "task_b": "Task B: price tier"}
    for key, title in titles.items():
        lines += [f"## {title}, test split", ""] + _markdown_table(results["test"][key])
        lines += [f"## {title}, {results['cv']['k']}-fold cross-validation (mean ± std)", ""]
        lines += _markdown_table(results["cv"][key])
    if results["skipped"]:
        lines += ["Not implemented yet, skipped: " + "; ".join(results["skipped"]) + ".", ""]
    if results["diagnostics"]["svm"]:
        lines += ["## Linear SVM: margin and support vectors", ""]
        lines += _markdown_table(results["diagnostics"]["svm"])
    for key, tables in results["errors"].items():
        for table_name, table in tables.items():
            lines += [f"## Mean test error by {table_name.replace('_', ' ')}, {titles[key]}", ""]
            lines += _markdown_table(table, first_column="Group")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# --- The pipeline ------------------------------------------------------------
def run_pipeline(X, y_price, feature_names, rows: pd.DataFrame | None = None,
                 config: dict | None = None, out_dir: str = OUT_DIR, fig_dir: str = FIG_DIR,
                 model_classes=None, tuning_run=None, label: str = "",
                 k: int = benchmark.N_FOLDS, table_dir: str | None = None) -> dict:
    """
    Stages 1 (split) to 8 on a ready feature table. Returns every number in one dict.

    rows:          optional table aligned with X, with "category" and "district"
                   columns, used only by the error analysis.
    model_classes: (tree class, SVM class) to use instead of the team's own.
    tuning_run:    the tuning study's function, or None to keep the given settings.
    label:         a prefix for figure titles (e.g. to mark a test run).
    table_dir:     where the LaTeX tables go (default: a tables/ folder inside out_dir).
    """
    X = np.asarray(X, dtype=float)
    y_price = np.asarray(y_price, dtype=float)
    feature_names = list(feature_names)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    config = merge_config(benchmark.CONFIG, config)
    figures: list[str] = []

    def figure_path(name: str) -> str:
        figures.append(os.path.join(fig_dir, name + ".png"))
        return figures[-1]

    # 1. Split ---------------------------------------------------------------
    train, val, test = split_rows(y_price)
    fit_rows = np.sort(np.concatenate([train, val]))
    threshold = float(np.median(y_price[train]))                    # training prices only
    tier, _ = data_prep.make_tier_label(y_price, threshold)
    print("\n== 1. Data and split ==")
    print(f"{len(X):,} listings, {X.shape[1]} features. "
          f"Train / validation / test: {len(train):,} / {len(val):,} / {len(test):,}.")
    print(f"Premium threshold (median of training prices): {threshold:,.0f} AZN. Premium share: "
          f"train {tier[train].mean():.3f}, validation {tier[val].mean():.3f}, "
          f"test {tier[test].mean():.3f}.")

    # 3. Tuning slot: sees train and validation, never test ---------------------
    print("\n== 3. Tuning ==")
    if tuning_run is None:
        print("Tuning study not provided yet (src/tuning.py): using the placeholder settings.")
    else:
        chosen = tuning_run(X[train], y_price[train], X[val], y_price[val], feature_names, fig_dir)
        config = merge_config(config, chosen)
        print(f"Settings chosen on the validation split: {chosen}")
    log_price = config["log_price"]

    # 4. Cross-validated benchmark on train + validation ------------------------
    print("\n== 4. Benchmark against scikit-learn (train + validation rows) ==")
    cv = benchmark.run(X[fit_rows], y_price[fit_rows], feature_names, config, k,
                       model_classes=model_classes)
    cv["k"] = k

    # 5-7. Final test evaluation, diagnostics, error analysis -------------------
    results = {
        "data": {"n": len(X), "n_features": X.shape[1], "train": len(train), "val": len(val),
                 "test": len(test), "threshold": threshold,
                 "premium_share": {"train": float(tier[train].mean()),
                                   "val": float(tier[val].mean()),
                                   "test": float(tier[test].mean())}},
        "config": config, "tuned": tuning_run is not None, "cv": cv,
        "test": {"task_a": {}, "task_b": {}}, "skipped": [],
        "diagnostics": {"confusion_matrix": {}, "svm": {}, "top_features": {}},
        "errors": {}, "worst_misses": {},
    }
    price_test, price_train = y_price[test], y_price[train]
    rows_test = rows.iloc[test].reset_index(drop=True) if rows is not None else None
    rows_train = rows.iloc[train] if rows is not None else None

    for task, key in TASKS.items():
        if task == "classification":
            y_fit, y_test = tier[fit_rows], tier[test]
        elif log_price:
            y_fit, y_test = np.log(y_price[fit_rows]), np.log(price_test)
        else:
            y_fit, y_test = y_price[fit_rows], price_test

        error_by_model, roc_curves = {}, {}
        for spec in benchmark.model_specs(task, config, model_classes=model_classes):
            name = spec["name"]
            fitted = fit_for_test(spec, task, X[fit_rows], y_fit, X[test])
            if fitted is None:
                results["skipped"].append(f"{name} ({key})")
                continue
            model, pred, scores = fitted["model"], fitted["pred"], fitted["scores"]
            title = f"{label}{name}"

            if task == "classification":
                results["test"][key][name] = evaluate.classification_metrics(y_test, pred, scores)
                matrix = evaluate.confusion_matrix(y_test, pred)
                results["diagnostics"]["confusion_matrix"][name] = matrix.tolist()
                report_figures.confusion_matrix_plot(
                    matrix, f"{title}: confusion matrix, test split",
                    figure_path(f"confusion_{slug(name)}"))
                if scores is not None:
                    roc_curves[name] = (*evaluate.roc_curve(y_test, scores),
                                        evaluate.roc_auc(y_test, scores))
                error_by_model[name] = (pred != y_test).astype(float)   # 1 = wrong tier
            else:
                price_pred = np.exp(pred) if log_price else pred
                if log_price:
                    results["test"][key][name] = {
                        "RMSE (log)": evaluate.rmse(y_test, pred),
                        "MAE (log)": evaluate.mae(y_test, pred),
                        "R2 (log)": evaluate.r2(y_test, pred),
                        "MAE (AZN)": evaluate.mae(price_test, price_pred)}
                else:
                    results["test"][key][name] = evaluate.regression_metrics(y_test, pred)
                report_figures.predicted_vs_actual_plot(
                    price_test, price_pred, f"{title}: predicted vs actual price, test split",
                    figure_path(f"predicted_vs_actual_{slug(name)}"))
                error_by_model[name] = np.abs(y_test - pred)
                shown = rows_test if rows_test is not None else pd.DataFrame(index=range(len(test)))
                results["worst_misses"][name] = error_analysis.worst_misses(
                    shown, price_test, price_pred).to_dict("records")

            # Optional outputs: each is skipped if the model does not have it.
            importances = model_api.feature_importances(model)
            if importances is not None and len(importances) == len(feature_names):
                order = np.argsort(-importances)[:10]
                results["diagnostics"]["top_features"][f"{name} ({key})"] = {
                    feature_names[i]: float(importances[i]) for i in order}
                report_figures.top_bars_plot(
                    feature_names, importances, f"{title}: feature importance ({key})",
                    "Share of total impurity decrease",
                    figure_path(f"importances_{slug(name)}_{key}"))

            svm = svm_diagnostics(model, fitted["X_fit"], y_fit) if task == "classification" else None
            if svm is not None:
                results["diagnostics"]["svm"][name] = svm
                weights = model_api.weights_and_bias(model)[0]
                if len(weights) == len(feature_names):
                    report_figures.top_bars_plot(
                        feature_names, weights, f"{title}: largest weights (standardized features)",
                        "Weight (positive pushes towards premium)",
                        figure_path(f"svm_weights_{slug(name)}"))
                history = model_api.objective_history(model)
                if history is not None:
                    report_figures.objective_plot(
                        *history, f"{title}: training objective", figure_path(f"objective_{slug(name)}"))

        if roc_curves:
            report_figures.roc_plot(roc_curves, f"{label}ROC curves, test split",
                                    figure_path("roc_curves"))

        tables = error_tables(error_by_model, price_test, price_train, rows_test, rows_train)
        results["errors"][key] = {name: table.to_dict("index") for name, table in tables.items()}
        for table_rows in results["errors"][key].values():
            for row in table_rows.values():
                row["Listings"] = int(row["Listings"])              # a count, not a float
        ylabel = "Share of wrong tier calls" if task == "classification" else (
            "Mean absolute error" + (" (log price)" if log_price else " (AZN)"))
        for table_name, table in tables.items():
            report_figures.error_by_group_plot(
                {model: table[model].to_dict() for model in error_by_model},
                f"{label}Test error by {table_name.replace('_', ' ')} ({key})", ylabel,
                figure_path(f"error_by_{table_name}_{key}"))

        print(f"\n== 5. Final evaluation on the test split, used once: {key} ==")
        evaluate.compare(results["test"][key])

    if results["diagnostics"]["svm"]:
        print("\n== 6. Linear SVM: margin and support vectors (train + validation rows) ==")
        evaluate.compare(results["diagnostics"]["svm"])
    for key, tables in results["errors"].items():
        for table_name, table in tables.items():
            print(f"\n== 7. Mean test error by {table_name.replace('_', ' ')} ({key}) ==")
            evaluate.compare(table)
    if results["skipped"]:
        print("\nNot implemented yet, skipped: " + "; ".join(results["skipped"]))

    # 8. Save -------------------------------------------------------------------
    results["figures"] = figures
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False,
                  default=lambda value: value.item() if hasattr(value, "item") else str(value))
    write_summary(results, os.path.join(out_dir, "summary.md"), label)
    table_dir = table_dir or os.path.join(out_dir, "tables")
    results["tables"] = report_tables.write_tables(results, table_dir)
    print(f"\n== 8. Saved ==\n{out_dir}/results.json, {out_dir}/summary.md, "
          f"{len(figures)} figures in {fig_dir}/, {len(results['tables'])} LaTeX tables in {table_dir}/")
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce every number and figure.")
    parser.add_argument("--synthetic", action="store_true",
                        help="use seeded made-up listings instead of the real data")
    args = parser.parse_args(argv)

    out_dir, fig_dir, table_dir, label = OUT_DIR, FIG_DIR, TABLE_DIR, ""
    if args.synthetic:
        print("SYNTHETIC DATA: checks the pipeline machinery only. Not project results.")
        X, y_price, feature_names = synthetic.make_listings()
        rows = None
        out_dir, label = os.path.join(OUT_DIR, "synthetic"), "SYNTHETIC DATA. "
    else:
        X, y_price, feature_names, table = benchmark.load_real_data(with_table=True)
        rows = pd.DataFrame({"category": table["Kateqoriya"].to_numpy(),
                             "district": table["location"].to_numpy(),
                             "area_m2": table["Sahə"].to_numpy()})
        print("\n== 2. EDA figures ==")
        from . import eda_plots                     # imported here: it creates report/figures
        eda_plots.generate_eda_figures()
    if out_dir != OUT_DIR:
        fig_dir = os.path.join(out_dir, "figures")  # test runs never touch the report's figures
        table_dir = os.path.join(out_dir, "tables")  # ... or its tables

    run_pipeline(X, y_price, feature_names, rows, out_dir=out_dir, fig_dir=fig_dir,
                 tuning_run=load_tuning(), label=label, table_dir=table_dir)

    # Member 3's wider SVM studies (learning-rate schedules, epochs, batch size, random
    # Fourier features) draw report/figures/svm_*.png. They run on her own feature set
    # and split, so their numbers are reported as a separate study, not in the tables above.
    if not args.synthetic and importlib.util.find_spec("src.svm_experiments"):
        print("\n== 9. SVM studies (src/svm_experiments.py) ==")
        importlib.import_module("src.svm_experiments").main()


if __name__ == "__main__":
    main()
