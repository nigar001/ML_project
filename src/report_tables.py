"""
report_tables.py — writes the report's tables as LaTeX, straight from the results.

report/report.tex pulls these files in with \\input{tables/...}, so every number
in the report's tables comes from the pipeline run and none is typed by hand.
numbers.tex holds single values (row counts, the tier threshold) as LaTeX
commands for use inside sentences.

A model that is not implemented yet gets a row saying so, so a missing model is
visible in the PDF instead of silently absent.
"""

from __future__ import annotations

import os

# Property categories as they appear in the data, in English for the report.
CATEGORY_NAMES = {
    "Yeni tikili": "New-build flat",
    "Köhnə tikili": "Old-build flat",
    "Həyət evi/Bağ evi": "House",
    "Torpaq": "Land",
    "Obyekt": "Commercial",
    "Ofis": "Office",
    "Qaraj": "Garage",
}

# Short model names for narrow tables.
SHORT_NAMES = {
    "Ours: tree": "Our tree",
    "Ours: SVM (Pegasos)": "Our SVM",
    "sklearn: DecisionTreeRegressor": "sk tree",
    "sklearn: DecisionTreeClassifier": "sk tree",
    "sklearn: SGDClassifier": "sk SGD",
    "sklearn: LinearSVC": "sk LinearSVC",
}

TASK_KEYS = ["task_a", "task_b"]
CV_HEADLINE = {"task_a": ["R2 (log)", "R2"], "task_b": ["F1"]}     # the CV column of each table
HEADERS = {"RMSE (log)": "RMSE", "MAE (log)": "MAE", "R2 (log)": "$R^2$", "R2": "$R^2$",
           "ROC-AUC": "AUC", "MAE (AZN)": "MAE (AZN)"}
ERROR_TABLE_TITLES = {"price_band": "By price band (training quantiles)",
                      "category": "By property category",
                      "district_size": "By training listings in the district"}


# --- Formatting -------------------------------------------------------------
def tex(text) -> str:
    """Escape the characters LaTeX treats specially in ordinary text."""
    text = str(text)
    for char in "&%#_":
        text = text.replace(char, "\\" + char)
    return text


def num(value) -> str:
    """A count or a price with thousands separators; a score with three decimals."""
    if isinstance(value, bool) or value is None:
        return "---"
    if isinstance(value, int) or abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", "{,}")
    return f"{value:.3f}"


def pm(pair) -> str:
    """A (mean, std) pair as 'mean ± std'."""
    return f"{num(pair[0])} $\\pm$ {num(pair[1])}"


def tabular(header: list[str], rows: list, align: str) -> str:
    """A booktabs table body. A row that is a plain string is written as it is (rules, spans)."""
    lines = [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [row if isinstance(row, str) else " & ".join(row) + " \\\\" for row in rows]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def _skipped(results: dict, key: str) -> list[str]:
    """Names of the models skipped for one task ('Ours: tree (task_a)' -> 'Ours: tree')."""
    suffix = f" ({key})"
    return [name[: -len(suffix)] for name in results["skipped"] if name.endswith(suffix)]


# --- The tables -------------------------------------------------------------
def numbers(results: dict) -> str:
    data, share = results["data"], results["data"]["premium_share"]
    values = {
        "NumListings": num(data["n"]), "NumFeatures": num(data["n_features"]),
        "NumTrain": num(data["train"]), "NumVal": num(data["val"]), "NumTest": num(data["test"]),
        "TierThreshold": num(int(round(data["threshold"]))),
        "PremiumShareTrain": f"{100 * share['train']:.1f}",
        "PremiumShareTest": f"{100 * share['test']:.1f}",
        "NumFolds": str(results["cv"]["k"]),
        "SettingsSource": "the validation study" if results["tuned"] else "placeholder values",
    }
    return "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())


def data_summary(results: dict) -> str:
    data, share = results["data"], results["data"]["premium_share"]
    rows = [
        ["Listings after cleaning", num(data["n"])],
        ["Features", num(data["n_features"])],
        ["Train / validation / test", f"{num(data['train'])} / {num(data['val'])} / {num(data['test'])}"],
        ["Tier threshold (AZN)", num(int(round(data["threshold"])))],
        ["Premium share, train / val / test (\\%)",
         " / ".join(f"{100 * share[part]:.1f}" for part in ("train", "val", "test"))],
    ]
    return tabular(["Property", "Value"], rows, "lr")


def settings(results: dict) -> str:
    config = results["config"]
    rows = [[f"Tree: \\texttt{{{tex(name)}}}", tex(value)] for name, value in config["tree"].items()]
    rows += [["Tree: criterion (Task B)", tex(config["tree_criterion"])],
             ["SVM: $\\lambda$", f"{config['svm']['lambda_']:g}"],
             ["Task A target", "log(price)" if config["log_price"] else "price"]]
    return tabular(["Setting", "Value"], rows, "lr")


def results_table(results: dict, key: str) -> str:
    """Test-split metrics of every model for one task, plus the cross-validated headline score."""
    test, cv = results["test"][key], results["cv"][key]
    rows: list
    metrics = list(next(iter(test.values()))) if test else []
    headline = next((m for m in CV_HEADLINE[key] if m in metrics), None)
    header = ["Model"] + [HEADERS.get(m, m) for m in metrics]
    if headline:
        header.append(f"CV {HEADERS.get(headline, headline)}")

    rows = [f"{tex(name)} & \\multicolumn{{{len(header) - 1}}}{{c}}{{not implemented yet}} \\\\"
            for name in _skipped(results, key)]
    for name, scores in test.items():
        row = [tex(name)] + [num(scores[m]) for m in metrics]
        if headline:
            row.append(pm(cv[name][headline]) if name in cv else "---")
        rows.append(row)
    return tabular(header, rows, "l" + "r" * (len(header) - 1))


def timing(results: dict) -> str:
    """Fit and predict seconds of each model at each training size."""
    timings = results["cv"]["timing"]
    if not timings:
        return tabular(["Model", "Time"], [["---", "---"]], "lr")
    columns = list(next(iter(timings.values())))
    header = ["Model"] + [tex(c.replace(" s @", " @ ").capitalize()) for c in columns]
    rows = [[tex(name)] + [f"{seconds[c]:.4f}" for c in columns] for name, seconds in timings.items()]
    return tabular(header, rows, "l" + "r" * len(columns))


def svm_margin(results: dict) -> str:
    rows: list = [[tex(name), num(d["||w||"]), num(d["Margin 2/||w||"]), num(d["Support vectors"]),
             f"{100 * d['Support vectors'] / d['Training points']:.1f}"]
            for name, d in results["diagnostics"]["svm"].items()]
    rows = [f"{tex(name)} & \\multicolumn{{4}}{{c}}{{not implemented yet}} \\\\"
            for name in _skipped(results, "task_b") if "SVM" in name] + rows
    header = ["Model", "$\\lVert\\mathbf{w}\\rVert$", "$2/\\lVert\\mathbf{w}\\rVert$",
              "Support vectors", "\\% of points"]
    return tabular(header, rows, "lrrrr")


def errors(results: dict, key: str) -> str:
    """Mean test error by kind of listing: the error-analysis tables for one task, stacked."""
    tables = results["errors"][key]
    first_table = next(iter(tables.values()), {})
    first_group = next(iter(first_table.values()), {})
    models = [name for name in first_group if name != "Listings"]
    header = ["Group", "$n$"] + [tex(SHORT_NAMES.get(m, m)) for m in models]
    rows: list = []
    for table_name, groups in tables.items():
        if rows:
            rows.append("\\midrule")
        title = ERROR_TABLE_TITLES.get(table_name, table_name)
        rows.append(f"\\multicolumn{{{len(header)}}}{{@{{}}l}}{{\\textit{{{tex(title)}}}}} \\\\")
        for group, values in groups.items():
            rows.append([tex(CATEGORY_NAMES.get(group, group)), num(values["Listings"])]
                        + [num(values[m]) for m in models])
    return tabular(header, rows, "lr" + "r" * len(models))


def agreement(results: dict) -> str:
    """Our model against its library twin on one held-out fold."""
    rows = []
    for pair, report in results["cv"]["agreement"].items():
        ours, theirs = [value for value in report.values() if isinstance(value, dict)]
        if "Depth" in ours:
            describe = lambda d: f"depth {d['Depth']}, {num(d['Leaves'])} leaves"
            same_root = (ours["Root feature"] == theirs["Root feature"]
                         # scikit-learn keeps thresholds as 32-bit floats: compare to that precision
                         and abs(ours["Root threshold"] - theirs["Root threshold"])
                         <= 1e-5 * max(1.0, abs(theirs["Root threshold"])))
            check = "same root split" if same_root else "different root split"
        else:
            describe = lambda d: f"$J$ = {d['Objective J']:.4f}, $\\lVert\\mathbf{{w}}\\rVert$ = {d['||w||']:.3f}"
            check = ""
        if "Same prediction %" in report:
            match = f"{report['Same prediction %']:.1f}\\% same calls"
        else:
            match = f"mean gap {report['Mean |gap|']:.4f}"
        task = "A" if "(task_a)" in pair else "B"
        kind = "Tree" if "Depth" in ours else "SVM"
        rows.append([f"{kind}, Task {task}", ", ".join(part for part in (match, check) if part),
                     describe(ours), describe(theirs)])
    if not rows:
        rows = ["\\multicolumn{4}{c}{not available: our models are not implemented yet} \\\\"]
    return tabular(["Pair", "Predictions", "Ours", "scikit-learn"], rows, "llll")


def worst_misses(results: dict) -> str:
    """The largest price errors of the first Task A model in the table."""
    if not results["worst_misses"]:
        return tabular(["Actual", "Predicted"], [["---", "---"]], "rr")
    name, misses = next(iter(results["worst_misses"].items()))
    # Listing details are shown when the pipeline was given them.
    header, rows = [], [[] for _ in misses]
    if "category" in misses[0]:
        header.append("Category")
        for row, miss in zip(rows, misses):
            row.append(tex(CATEGORY_NAMES.get(miss["category"], miss["category"])))
    if "area_m2" in misses[0]:
        header.append("Area (m$^2$)")
        for row, miss in zip(rows, misses):
            row.append(num(int(round(miss["area_m2"]))))
    header += ["Actual (AZN)", "Predicted (AZN)"]
    for row, miss in zip(rows, misses):
        row += [num(int(round(miss["Actual"]))), num(int(round(miss["Predicted"])))]
    align = "l" * ("category" in misses[0]) + "r" * (len(header) - ("category" in misses[0]))
    return f"% model: {name}\n" + tabular(header, rows, align)


def write_tables(results: dict, table_dir: str) -> list[str]:
    """Write every table file into table_dir and return the paths."""
    os.makedirs(table_dir, exist_ok=True)
    files = {
        "numbers": numbers(results), "data_summary": data_summary(results),
        "settings": settings(results), "timing": timing(results),
        "svm_margin": svm_margin(results), "agreement": agreement(results),
        "worst_misses": worst_misses(results),
    }
    for key in TASK_KEYS:
        files[f"results_{key}"] = results_table(results, key)
        files[f"errors_{key}"] = errors(results, key)
    paths = []
    for name, content in files.items():
        paths.append(os.path.join(table_dir, name + ".tex"))
        with open(paths[-1], "w", encoding="utf-8") as handle:
            handle.write("% Generated by python -m src.run_all. Do not edit by hand.\n" + content)
    return paths
