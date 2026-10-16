"""
report_figures.py — the result figures for the report (matplotlib only).

Each function draws one figure, saves it as a PNG and returns the path.
Every figure has a title and labelled axes. One style throughout.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#2e9e6b", "#8e5bd0", "#8a8f98"]
DPI = 200


def _axes(title: str, xlabel: str, ylabel: str, size=(6.0, 4.0)):
    fig, ax = plt.subplots(figsize=size)
    # The title belongs to the whole figure, so long axis labels cannot push it off the edge.
    fig.suptitle(textwrap.fill(title, width=int(size[0] * 9)), fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")      # keeps a legend placed outside the axes
    plt.close(fig)
    return path


def confusion_matrix_plot(matrix, title: str, path: str) -> str:
    """The 2x2 table [[TN, FP], [FN, TP]] as a coloured grid with the counts written in."""
    matrix = np.asarray(matrix)
    fig, ax = _axes(title, "Predicted tier", "Actual tier", size=(4.2, 3.8))
    ax.grid(False)
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1], ["standard", "premium"])
    ax.set_yticks([0, 1], ["standard", "premium"])
    for i in range(2):
        for j in range(2):
            dark = matrix[i, j] > matrix.max() / 2
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if dark else "black")
    return _save(fig, path)


def roc_plot(curves: dict, title: str, path: str) -> str:
    """ROC curves of several models. `curves` is {model name: (fpr, tpr, auc)}."""
    fig, ax = _axes(title, "False positive rate", "True positive rate (recall)", size=(5.2, 4.6))
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=2,
                label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="random guess")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def top_bars_plot(names, values, title: str, xlabel: str, path: str, top: int = 15) -> str:
    """Horizontal bars for the `top` largest values by size. Negative values are drawn in orange."""
    values = np.asarray(values, dtype=float)
    keep = np.argsort(-np.abs(values))[:top][::-1]                  # largest at the top
    fig, ax = _axes(title, xlabel, "", size=(6.0, 0.3 * len(keep) + 1.4))
    colors = [COLORS[0] if values[i] >= 0 else COLORS[1] for i in keep]
    ax.barh([str(names[i]) for i in keep], values[keep], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    return _save(fig, path)


def objective_plot(steps, values, title: str, path: str) -> str:
    """The SVM training objective J against the training step: it should go down and flatten."""
    fig, ax = _axes(title, "Training step", "Objective J(w, b)")
    ax.plot(np.asarray(steps, float), np.asarray(values, float), color=COLORS[0], linewidth=2,
            marker="o", markersize=3)
    ax.xaxis.get_major_locator().set_params(integer=True)
    return _save(fig, path)


def predicted_vs_actual_plot(price_true, price_pred, title: str, path: str) -> str:
    """Predicted against actual price, both on log axes. Points on the diagonal are exact."""
    price_true, price_pred = np.asarray(price_true, float), np.asarray(price_pred, float)
    fig, ax = _axes(title, "Actual price (AZN)", "Predicted price (AZN)", size=(5.0, 4.6))
    ax.scatter(price_true, price_pred, s=4, alpha=0.25, color=COLORS[0])
    low, high = price_true.min(), price_true.max()
    ax.plot([low, high], [low, high], color="black", linewidth=1, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return _save(fig, path)


def error_by_group_plot(mean_error: dict, title: str, ylabel: str, path: str) -> str:
    """
    Grouped bars: one group per kind of listing, one bar per model.
    `mean_error` is {model name: {group: mean error}}; all models share the groups.
    """
    groups = list(next(iter(mean_error.values())))
    fig, ax = _axes(title, "", ylabel, size=(max(6.0, 1.1 * len(groups) + 2), 4.0))
    width = 0.8 / len(mean_error)
    for i, (name, by_group) in enumerate(mean_error.items()):
        positions = np.arange(len(groups)) + i * width
        ax.bar(positions, [by_group[g] for g in groups], width=width * 0.92,
               color=COLORS[i % len(COLORS)], label=name)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2, [str(g) for g in groups],
                  rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))   # beside the bars
    return _save(fig, path)
