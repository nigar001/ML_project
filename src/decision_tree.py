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
        self.feature_importances_ = None   # set by fit(): share of total impurity decrease
        self._rng = np.random.default_rng(random_state)

    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""

        # In simple words, it measures how mixed the target labels are in a node
        # For Gini (btw we want to keep it as small as possible just because we want the best split which is very intuitive)

        # I(G) = 1 - sum(math.pow(p_i, 2))
        # So low impurity means the node data is purer
        # So high impurity means the node data is less pure


        # From information theory
        # We can also calculate how unexpected will be the target with probability p in the dataset(node)
        # For that we calculate I(x_target) = -log_2base(P_x_target), so if the data contains 95 percent of true and 5 percent of false, if randomly chosen element was false we will be surprised more so I(False) will be more since it has lower probability, in other words lower chance of beging selected


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
                return float(-1 * (np.sum(class_probabilities * np.log2(class_probabilities))))

            else:
                raise ValueError(
                    f"Unknown classification criterion: {self.criterion}"
                )
        elif self.task == "regression":
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


    def _impurity_of_counts(self, counts, sizes):
        """Gini / entropy of many nodes at once. counts: (m, k) class counts, sizes: (m,)."""
        p = counts / sizes[:, None]
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=1)
        # entropy: treat 0 * log(0) as 0
        safe = np.where(p > 0, p, 1.0)
        return -np.sum(p * np.log2(safe), axis=1)

    def _best_split(self, X, y):
        """
        Find the (feature, threshold, impurity decrease) with the largest gain, or None.

        Same objective as _candidate_thresholds + _impurity_decrease (kept above as the
        simple reference version), but instead of re-scanning the node for every threshold
        (O(n^2) per feature) we sort each feature once and get the left/right statistics
        of EVERY threshold from cumulative sums (O(n log n) per feature):
            classification: cumulative class counts -> Gini / entropy of both children
            regression:     cumulative sum(y) and sum(y^2) -> variance of both children
        """
        n_samples, n_features = X.shape

        if n_samples < 2:
            return None

        parent_impurity = self._impurity(y)

        if self.task == "classification":
            # One-hot class indicators; cumulative sums over the sorted rows = class counts.
            _, class_index = np.unique(y, return_inverse=True)
            onehot = np.eye(class_index.max() + 1)[class_index]
            total_counts = onehot.sum(axis=0)
        else:
            # Variance does not change if y is shifted; centering keeps the sums accurate.
            y_centered = np.asarray(y, dtype=float) - np.mean(y)
            total_sum = y_centered.sum()
            total_sq = np.sum(y_centered ** 2)

        # Size of the left child for the split placed after sorted row i (i = 0 .. n-2).
        n_left = np.arange(1, n_samples, dtype=float)
        n_right = n_samples - n_left
        big_enough = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)

        best_split = None
        best_decrease = -np.inf

        # Random feature order: it only decides ties between equally good splits.
        for feature in self._rng.permutation(n_features):
            column = X[:, feature]
            order = np.argsort(column, kind="stable")
            sorted_values = column[order]

            # A threshold can only sit between two DIFFERENT neighbouring values.
            valid = (sorted_values[:-1] < sorted_values[1:]) & big_enough
            if not valid.any():
                continue

            if self.task == "classification":
                left_counts = np.cumsum(onehot[order], axis=0)[:-1]
                right_counts = total_counts - left_counts
                children_impurity = (
                    n_left * self._impurity_of_counts(left_counts, n_left)
                    + n_right * self._impurity_of_counts(right_counts, n_right)
                ) / n_samples
            else:
                ys = y_centered[order]
                left_sum = np.cumsum(ys)[:-1]
                left_sq = np.cumsum(ys ** 2)[:-1]
                # n * variance = sum(y^2) - sum(y)^2 / n, for each child
                left_ss = left_sq - left_sum ** 2 / n_left
                right_ss = (total_sq - left_sq) - (total_sum - left_sum) ** 2 / n_right
                children_impurity = (left_ss + right_ss) / n_samples

            decrease = np.where(valid, parent_impurity - children_impurity, -np.inf)
            i = int(np.argmax(decrease))           # first best threshold on this feature

            if decrease[i] > best_decrease:
                low, high = sorted_values[i], sorted_values[i + 1]
                threshold = (low + high) / 2.0
                if threshold >= high:
                    # Float rounding: the midpoint of two neighbouring floats can equal the
                    # upper one, which would send both values left. Use the lower value.
                    threshold = low
                best_decrease = float(decrease[i])
                best_split = (int(feature), float(threshold), best_decrease)

        return best_split

    def _leaf_value(self, y):
        """
        Classification:
            Return the probability of each class.

        Regression:
            Return the mean target value.
        """
        """
            What a leaf predicts.
            Classification: the fraction of each class in the leaf, in the
                order of self.classes_. E.g. 3 'low' and 1 'high' -> [0.75, 0.25].
                predict() takes the most likely class; predict_proba() returns this.
            Regression: the mean of y. The mean is the single number that
                minimizes squared error, which is exactly what MSE measures.
        """
        if self.task == "classification":
            # Finding the count of each class
            counts = np.array(
                [
                    np.sum(y == c)
                    for c in self.classes_
                ],
                dtype=float,
            )
            return counts / counts.sum()

        elif self.task == "regression":
            return float(np.mean(y))

    def _make_leaf(self, y) -> Node:
        leaf = Node()
        leaf.value = self._leaf_value(y)
        return leaf

    def _grow(self, X, y, depth: int) -> Node:
        """
        Recursively build the tree.

        At each node:
          1. Check the stopping rules. If any fires, return a leaf.
          2. Otherwise find the best split.
          3. If no split is good enough, return a leaf.
          4. Otherwise split the data and grow the two children (depth + 1).
        """
        n_samples = len(y)

        # --- Stopping rules (these control overfitting) ---
        reached_max_depth = self.max_depth is not None and depth >= self.max_depth
        too_few_samples = n_samples < self.min_samples_split
        already_pure = self._impurity(y) == 0.0  # nothing left to improve

        if reached_max_depth or too_few_samples or already_pure:
            return self._make_leaf(y)

        # --- Find the best question to ask here ---
        split = self._best_split(X, y)

        # No valid split (e.g. min_samples_leaf rules everything out,
        # or all features are constant) -> leaf.
        if split is None:
            return self._make_leaf(y)

        feature, threshold, decrease = split

        # The best split doesn't improve purity enough to be worth it -> leaf.
        # Like scikit-learn, the decrease is weighted by this node's share of ALL training
        # rows, so one threshold means the same thing at the top and at the bottom of the tree.
        weighted_decrease = decrease * n_samples / self._n_train
        if weighted_decrease < self.min_impurity_decrease:
            return self._make_leaf(y)

        # Bookkeeping for feature_importances_ (total impurity decrease per feature).
        self._importance_sums[feature] += weighted_decrease

        # --- Split and recurse ---
        goes_left = X[:, feature] <= threshold

        node = Node()
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[goes_left], y[goes_left], depth + 1)
        node.right = self._grow(X[~goes_left], y[~goes_left], depth + 1)
        return node

    def fit(self, X, y):

        # 1. Convert input data into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # 2. Check that X is a table (2D array)
        if X.ndim != 2:
            raise ValueError("X must have rows and columns")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")

        # 3. Check that every sample has a target value
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        # 4. Check that the dataset is not empty
        if len(y) == 0:
            raise ValueError("Cannot fit on an empty dataset")

        # 5. Remember how many features each sample has
        number_of_features = X.shape[1]
        self.n_features_in_ = number_of_features

        # 6. Remember the possible classes for classification
        if self.task == "classification":
            self.classes_ = np.unique(y)

        # 7. Prepare the random number generator
        self._rng = np.random.default_rng(self.random_state)

        # 8. Build the tree, starting at depth zero
        self._n_train = len(y)
        self._importance_sums = np.zeros(number_of_features)
        root_node = self._grow(X, y, depth=0)
        self.root = root_node

        # Impurity-decrease feature importance: each feature's share of the total decrease
        # (sums to 1; all zeros if the tree never split).
        total = self._importance_sums.sum()
        self.feature_importances_ = (
            self._importance_sums / total if total > 0 else self._importance_sums.copy()
        )

        # 9. Return the trained decision tree
        return self

    def _predict_one(self, x, node: Node):
        """
        Follow the questions until we reach a leaf:
            x[feature] <= threshold -> go left, otherwise go right.
        Returns the leaf's stored value.
        """
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _check_X(self, X):
        if self.root is None:
            raise RuntimeError("The tree is not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)  # allow a single sample
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the tree was fitted on "
                f"{self.n_features_in_}"
            )
        return X

    def predict(self, X):
        X = self._check_X(X)
        leaf_values = [self._predict_one(x, self.root) for x in X]

        if self.task == "classification":
            # Each leaf value is a probability vector; pick the most likely class.
            probabilities = np.vstack(leaf_values)
            return self.classes_[np.argmax(probabilities, axis=1)]

        return np.array(leaf_values, dtype=float)

    def predict_proba(self, X):
        #in other words a test sample come to that leaf, in that leaf the probabolity of target target variables given, and we are choosing the highest probability classficiation target, since it has higher probability

        """Classification only: return class probabilities from leaf counts."""
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        X = self._check_X(X)
        # Shape (n_samples, n_classes); columns follow the order of self.classes_.
        return np.vstack([self._predict_one(x, self.root) for x in X])