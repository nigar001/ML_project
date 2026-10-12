"""
error_analysis.py — where do the models fail? (Problem 6)

Small tables that split the test-set error by kind of listing: property
category, price band (the luxury tail) and how well a district is covered by
the training data (sparse districts), plus the single worst predictions.

Every function takes one error value per test listing:
  - Task A: the absolute error of the predicted (log) price,
  - Task B: 1 if the tier was predicted wrongly, else 0 (so the mean is the
    share of mistakes).
Band edges and district counts come from the TRAINING rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Price bands as quantiles of the training prices: cheap tail, lower and upper
# middle, expensive, and the luxury top 1%.
PRICE_QUANTILES = [0.10, 0.50, 0.90, 0.99]
PRICE_BAND_NAMES = ["bottom 10%", "10-50%", "50-90%", "90-99%", "top 1%"]

# A district with fewer training listings than this is "sparse".
DISTRICT_SIZE_EDGES = [50, 500]
DISTRICT_SIZE_NAMES = ["under 50", "50-499", "500 or more"]


def by_group(groups, error, order=None) -> pd.DataFrame:
    """
    Mean error per group. Returns a table indexed by group with the columns
    "Listings" and "Mean error", largest group first (or in `order` if given).
    """
    table = pd.DataFrame({"group": np.asarray(groups), "error": np.asarray(error, dtype=float)})
    result = table.groupby("group")["error"].agg(["size", "mean"])
    result.columns = ["Listings", "Mean error"]
    result.index.name = None
    if order is not None:
        return result.reindex([name for name in order if name in result.index])
    return result.sort_values("Listings", ascending=False, kind="stable")


def by_price_band(price, error, price_train) -> pd.DataFrame:
    """Mean error per price band; the band edges are quantiles of the training prices."""
    edges = np.quantile(np.asarray(price_train, dtype=float), PRICE_QUANTILES)
    band = np.searchsorted(edges, np.asarray(price, dtype=float), side="left")
    names = np.array(PRICE_BAND_NAMES)[band]
    return by_group(names, error, order=PRICE_BAND_NAMES)


def by_district_size(district, error, district_train) -> pd.DataFrame:
    """
    Mean error by how many TRAINING listings the listing's district has.
    A district that never appears in training counts as 0 listings.
    """
    train_counts = pd.Series(np.asarray(district_train)).value_counts()
    counts = pd.Series(np.asarray(district)).map(train_counts).fillna(0).to_numpy()
    band = np.searchsorted(DISTRICT_SIZE_EDGES, counts, side="right")
    names = np.array(DISTRICT_SIZE_NAMES)[band]
    return by_group(names, error, order=DISTRICT_SIZE_NAMES)


def worst_misses(rows: pd.DataFrame, y_true, y_pred, n: int = 10) -> pd.DataFrame:
    """
    The n listings with the largest absolute error. `rows` holds the columns
    to show for each listing (e.g. category, district, area); "Actual",
    "Predicted" and "Error" are added on the right.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    table = rows.reset_index(drop=True).copy()
    table["Actual"] = y_true
    table["Predicted"] = y_pred
    table["Error"] = np.abs(y_true - y_pred)
    return table.sort_values("Error", ascending=False, kind="stable").head(n)
