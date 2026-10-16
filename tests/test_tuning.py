"""
test_tuning.py — checks for the validation study in src/tuning.py.

Runs on small seeded synthetic listings; the tree path uses the stand-in tree.

Run from the repo root:
    python -m pytest tests/test_tuning.py -v
"""

from __future__ import annotations

import numpy as np

from src import synthetic, tuning
from tests.standins import StandInTree


class _TemplateTree:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        raise NotImplementedError("TODO")


def _split():
    X, price, names = synthetic.make_listings(n=500, seed=3)
    return X[:350], price[:350], X[350:], price[350:], names


def test_svm_lambda_is_chosen_from_the_sweep_and_tree_waits_for_its_code(tmp_path):
    X_tr, p_tr, X_val, p_val, names = _split()
    chosen = tuning.run(X_tr, p_tr, X_val, p_val, names, str(tmp_path), tree_class=_TemplateTree)

    assert chosen["svm"]["lambda_"] in tuning.SVM_LAMBDAS
    assert chosen["svm"]["n_epochs"] == tuning.SVM_EPOCHS
    assert "tree" not in chosen                                    # placeholders stay
    assert (tmp_path / "tuning_svm_lambda_f1.png").exists()
    assert (tmp_path / "tuning_svm_lambda_margin.png").exists()


def test_tree_settings_come_from_the_sweeps(tmp_path):
    X_tr, p_tr, X_val, p_val, names = _split()
    chosen = tuning.run(X_tr, p_tr, X_val, p_val, names, str(tmp_path), tree_class=StandInTree)

    assert chosen["tree"]["max_depth"] in tuning.TREE_DEPTHS
    assert chosen["tree"]["min_samples_leaf"] in tuning.TREE_MIN_LEAF
    assert chosen["tree_criterion"] in ("gini", "entropy")
    for name in ("tuning_tree_depth_task_a", "tuning_tree_depth_task_b", "tuning_tree_min_leaf"):
        assert (tmp_path / f"{name}.png").exists()


def test_the_same_data_gives_the_same_choice(tmp_path):
    X_tr, p_tr, X_val, p_val, names = _split()
    first = tuning.run(X_tr, p_tr, X_val, p_val, names, str(tmp_path), tree_class=StandInTree)
    second = tuning.run(X_tr, p_tr, X_val, p_val, names, str(tmp_path), tree_class=StandInTree)
    assert first == second
    assert np.isfinite(first["svm"]["lambda_"])
