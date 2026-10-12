"""
test_error_analysis.py — checks for the error tables in src/error_analysis.py.

Small examples whose answer is worked out by hand in the comments.

Run from the repo root:
    python -m pytest tests/test_error_analysis.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import error_analysis


def test_by_group_counts_and_averages():
    # flat: errors 1 and 3 -> mean 2;  land: error 10 -> mean 10
    table = error_analysis.by_group(["flat", "land", "flat"], [1.0, 10.0, 3.0])
    assert table.index.tolist() == ["flat", "land"]                 # largest group first
    assert table["Listings"].tolist() == [2, 1]
    assert table["Mean error"].tolist() == [2.0, 10.0]


def test_by_group_keeps_a_given_order_and_drops_empty_groups():
    table = error_analysis.by_group(["b", "a"], [1.0, 2.0], order=["a", "missing", "b"])
    assert table.index.tolist() == ["a", "b"]


def test_price_bands_use_training_quantiles_only():
    price_train = np.arange(1, 101, dtype=float)       # 1..100: edges near 10.9, 50.5, 90.1, 99.0
    price = np.array([5.0, 30.0, 70.0, 95.0, 100.0, 5000.0])
    error = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
    table = error_analysis.by_price_band(price, error, price_train)

    assert table.index.tolist() == error_analysis.PRICE_BAND_NAMES   # cheapest band first
    assert table["Listings"].tolist() == [1, 1, 1, 1, 2]
    assert table["Mean error"].tolist() == [1.0, 2.0, 3.0, 4.0, 6.0]  # top 1%: (5 + 7) / 2

    # The test prices do not move the edges: a very different test set, same bands.
    again = error_analysis.by_price_band(np.array([5.0, 5.0]), np.array([1.0, 3.0]), price_train)
    assert again.index.tolist() == ["bottom 10%"] and again["Mean error"].tolist() == [2.0]


def test_district_size_counts_training_listings():
    district_train = ["big"] * 600 + ["mid"] * 50 + ["small"] * 49
    district = ["big", "mid", "small", "never seen", "big"]
    error = [1.0, 2.0, 3.0, 5.0, 3.0]
    table = error_analysis.by_district_size(district, error, district_train)

    assert table.index.tolist() == error_analysis.DISTRICT_SIZE_NAMES
    assert table["Listings"].tolist() == [2, 1, 2]                   # small + never seen | mid | big
    assert table["Mean error"].tolist() == [4.0, 2.0, 2.0]


def test_worst_misses_are_sorted_by_absolute_error():
    rows = pd.DataFrame({"category": ["flat", "house", "land"]}, index=[10, 20, 30])
    table = error_analysis.worst_misses(rows, [100.0, 200.0, 300.0], [110.0, 150.0, 301.0], n=2)

    assert table["category"].tolist() == ["house", "flat"]           # errors 50 and 10
    assert table["Error"].tolist() == [50.0, 10.0]
    assert list(table.columns) == ["category", "Actual", "Predicted", "Error"]
