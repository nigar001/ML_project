"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        if self.left is None and self.right is None:
            return True
        return False


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")

        if task == "classification" and criterion not in ("gini", "entropy"):
            raise ValueError("Classification requires gini or entropy")

        if task == "regression" and criterion != "mse":
            raise ValueError("Regression requires criterion='mse'")

        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")

        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease cannot be negative")

        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        self.root = None
        self.classes_ = None
        self.n_features_in_ = None
        self._rng = np.random.default_rng(random_state)

    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""

        # In simple words, it measures how mixed the target labels are in a node
        # For Gini (btw we want to keep it as small as possible just because we want the best split which is very intuitive)

        # I(G) = 1 - sum(math.pow(p_i, 2))
        # So low impurity means the node data is purer
        # So high impurity means the node data is less pure


        #From information theory
        # we can also calculate how unexpected will be the target with probability p in the dataset(node)
        #for that we calculate I(x_target) = -log_2base(P_x_target), so if the data contains 95 percent of true and 5 percent of false, if randomly chosen element was false we will be surprised more so I(False) will be more since it has lower probability, in other words lower chance of beging selected


        if len(y) == 0:
            return 0.0

        if self.task == "classification":
            unique_classes, class_counts = np.unique(
                y,
                return_counts=True
            )
            total_samples = len(y)
            class_probabilities = class_counts / total_samples # which is gonna be a numpy array
            if self.criterion == "gini":
                return 1 - np.sum(np.power(class_probabilities, 2))

            elif self.criterion == "entropy":
                return -1 * (np.sum(class_probabilities * np.log2(class_probabilities)))

            else:
                raise ValueError(
                    f"Unknown classification criterion: {self.criterion}"
                )

        elif self.task == "regression":
            if self.criterion not in ("mse", "variance"):
                raise ValueError(
                    f"Unknown regression criterion: {self.criterion}"
                )
            mean_y = float(np.mean(y))
            return float(np.mean((y - mean_y) ** 2))

        else :
            raise ValueError("Unknown task")


    def _candidate_thresholds(self, column):
        """
        Generate possible split thresholds for one numerical feature.

        Example:
            column = [1, 3, 7]

            Candidate thresholds = [2, 5]

        These thresholds allow the tree to try different ways
        of dividing the samples into left and right groups.
        """
        #we are getting the uniqeue values and sort them in ascending order and choose one threshold to test the feature split impurity

        unique_values = np.unique(column)

        # If there is only one unique value this feature, which is very intuitive
        # cannot divide the samples into two non-empty groups
        # Therefore, there are no candidate thresholds
        if len(unique_values) < 2:
            return []

        midpoints = (
            unique_values[:-1] + unique_values[1:]
        ) / 2.0

        # Step 4: Randomize the order in which we try the thresholds.
        # This only affects which split is selected when multiple
        # candidates have exactly the same impurity decrease.
        random_order = self._rng.permutation(len(midpoints))

        return midpoints[random_order]


    def _impurity_decrease(
        self, column, y, threshold, parent_impurity
    ):
        """
        Calculate the gain from one proposed split.

        Gain = parent impurity - weighted child impurity

        Gain = I(parent) - [n_left/n * I(left) + n_right/n * I(right)]
        """

        n = len(y)

        # Samples satisfying the condition go left.
        # In other words, if I have f1 <= threshold my targets = [true, true, true, false, false]

        goes_left = column <= threshold

        n_left = int(goes_left.sum()) #since we have here true and false as classification it will be 3
        n_right = n - n_left

        # For example if a leaf has less than predefines min samples then stop
        if min(n_left, n_right) < self.min_samples_leaf:
            return None

        # Calculate each child's impurity.
        # bascially how pure the data is meant by impurity

        left_impurity = self._impurity(
            y[goes_left]
        )

        right_impurity = self._impurity(
            y[~goes_left]
        )

        # Weight each child by its proportion of samples.
        children_impurity = (
            (n_left / n) * left_impurity
            + (n_right / n) * right_impurity
        )

        # Positive gain means the split reduces impurity.
        # Basically we want to make the node data, after each feature split as pure as possbile, Like parent_data = [t, t, t, t, f, f, f, f]
        # As visible, it is not pure, but let's say we want to split regarding to a column and its some threshold we get left = [t,t,t,f] right = [t,f,f,f]
        # So the data is purer compared to compared to parent node data

        return parent_impurity - children_impurity


    def _best_split(self, X, y):
        # We are basically trying all features and its candidate thresholds

        n_samples, n_features = X.shape

        if n_samples < 2:
            return None
        # Of course we should check the parent data impurity to compare which feature and which throshold makes it better(with less impurity)

        parent_impurity = self._impurity(y)

        best_split = None
        best_decrease = -np.inf

        # Examine every feature.
        for feature in self._rng.permutation(n_features):

            # Select one feature column.
            column = X[:, feature]

            # Generate possible thresholds for this feature.
            for threshold in self._candidate_thresholds(column):

                decrease = self._impurity_decrease(
                    column,
                    y,
                    threshold,
                    parent_impurity,
                )

                if (
                    decrease is not None
                    and decrease > best_decrease
                ):

                    best_decrease = decrease

                    best_split = (
                        int(feature),
                        float(threshold),
                        float(decrease),
                    )
        return best_split


    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        raise NotImplementedError("TODO: predict_proba (classification)")




