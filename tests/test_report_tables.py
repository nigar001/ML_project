"""
test_report_tables.py — checks for the LaTeX formatting helpers in src/report_tables.py.

Run from the repo root:
    python -m pytest tests/test_report_tables.py -v
"""

from __future__ import annotations

from src import report_tables


def test_special_characters_are_escaped():
    assert report_tables.tex("bottom 10% & min_samples_leaf #1") == (
        "bottom 10\\% \\& min\\_samples\\_leaf \\#1")


def test_numbers_are_formatted_for_the_report():
    assert report_tables.num(64248) == "64{,}248"          # a count
    assert report_tables.num(90426.47) == "90{,}426"       # a price
    assert report_tables.num(0.73384) == "0.734"           # a score
    assert report_tables.num(None) == "---"
    assert report_tables.pm((0.7223, 0.0035)) == "0.722 $\\pm$ 0.004"


def test_tabular_layout():
    table = report_tables.tabular(["Model", "F1"], [["A", "0.900"], "\\midrule", ["B", "0.800"]], "lr")
    assert table.splitlines() == [
        "\\begin{tabular}{@{}lr@{}}", "\\toprule", "Model & F1 \\\\", "\\midrule",
        "A & 0.900 \\\\", "\\midrule", "B & 0.800 \\\\", "\\bottomrule", "\\end{tabular}"]


def test_categories_have_english_names():
    assert report_tables.CATEGORY_NAMES["Torpaq"] == "Land"
    assert len(set(report_tables.CATEGORY_NAMES.values())) == 7
