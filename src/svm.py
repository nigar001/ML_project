"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""






from __future__ import annotations
 
import numpy as np
 
 
class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,       # regularisation strength; roughly C = 1 / (n * lambda)
        n_iters: int = 20_000,       # number of sub-gradient STEPS (ignored if n_epochs is given)
        n_epochs: float | None = None,   # passes over the data; overrides n_iters
        batch_size: int = 32,        # 1 = classic Pegasos, n = full-batch sub-gradient descent
        schedule: str = "pegasos",   # "pegasos" 1/(lambda t) | "sqrt" eta0/sqrt(t) | "constant" eta0
        eta0: float = 1.0,           # used by "sqrt" and "constant"
        project: bool = True,        # projection onto ||w|| <= 1/sqrt(lambda) (from the Pegasos paper)
        record_every: int = 200,     # evaluate the full objective every this many steps
        random_state: int = 42,
    ):
        self.lambda_, self.n_iters, self.n_epochs = lambda_, n_iters, n_epochs
        self.batch_size, self.schedule, self.eta0 = batch_size, schedule, eta0
        self.project, self.record_every, self.random_state = project, record_every, random_state
        self.w, self.b = None, 0.0
 
    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_pm1(y, classes=None):
        """Map any 2-class labels (e.g. {0,1}) to {-1,+1}; the larger label becomes +1. Returns (ypm, classes)."""
        y = np.asarray(y)
        if classes is None:
            classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"PegasosSVM is binary; got classes {classes}")
        return np.where(y == classes[1], 1.0, -1.0), classes
 
    def _eta(self, t: int) -> float:
        if self.schedule == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.schedule == "sqrt":
            return self.eta0 / np.sqrt(t)
        if self.schedule == "constant":
            return self.eta0
        raise ValueError(f"unknown schedule {self.schedule!r}")
 
    def objective(self, Xa, ypm) -> float:
        """Primal objective J on augmented data Xa = [X, 1] (what Pegasos minimises)."""
        margins = ypm * (Xa @ self._wa)
        return 0.5 * self.lambda_ * float(self._wa @ self._wa) + float(np.mean(np.maximum(0.0, 1.0 - margins)))
 
    # ------------------------------------------------------------------ training
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf. Impute missing values (with TRAIN statistics) before fitting.")
        ypm, self.classes_ = self._to_pm1(y)
        n = len(X)
        Xa = np.hstack([X, np.ones((n, 1))])                  # fold the bias into the weights
        bs = min(self.batch_size, n)
        n_steps = self.n_iters if self.n_epochs is None else max(1, int(np.ceil(self.n_epochs * n / bs)))
        self.n_steps_, self.epochs_ = n_steps, n_steps * bs / n
 
        rng = np.random.default_rng(self.random_state)
        self._wa = np.zeros(Xa.shape[1])
        self.history_ = []                                    # (step, objective) -> convergence curve
        radius = 1.0 / np.sqrt(self.lambda_)
        for t in range(1, n_steps + 1):
            idx = rng.integers(0, n, bs)                      # sample a mini-batch with replacement
            eta = self._eta(t)
            violators = ypm[idx] * (Xa[idx] @ self._wa) < 1.0   # hinge is active only here
            subgrad = (ypm[idx][violators, None] * Xa[idx][violators]).sum(axis=0) / bs
            self._wa = (1.0 - eta * self.lambda_) * self._wa + eta * subgrad
            if self.project:
                nrm = np.linalg.norm(self._wa)
                if nrm > radius:
                    self._wa *= radius / nrm
            if t == 1 or t % self.record_every == 0 or t == n_steps:
                self.history_.append((t, self.objective(Xa, ypm)))
 
        self.w, self.b = self._wa[:-1].copy(), float(self._wa[-1])
        # --- margin diagnostics (Problem 3c) ---
        self.w_norm_ = float(np.linalg.norm(self.w))
        self.margin_width_ = 2.0 / self.w_norm_ if self.w_norm_ > 0 else np.inf
        self.train_margins_ = ypm * (X @ self.w + self.b)      # functional margins y_i (w.x_i + b)
        self.support_ = np.where(self.train_margins_ <= 1.0)[0]  # support vectors: y_i (w.x_i + b) <= 1
        return self
 
    # ------------------------------------------------------------------ inference
    def decision_function(self, X):
        """Signed score w.x + b (positive = classes_[1]); also usable as a ranking score for ROC-AUC."""
        return np.asarray(X, dtype=float) @ self.w + self.b
 
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])
 
    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
 
 
class RandomFourierFeatures:
    """
    Random Fourier features (Rahimi & Recht, 2007) approximating the RBF kernel
    k(x, x') = exp(-gamma * ||x - x'||^2):
 
        phi(x) = sqrt(2/D) * cos(W^T x + b),   W_ij ~ N(0, 2*gamma),   b_j ~ U[0, 2*pi]
 
    so that phi(x).phi(x') ~= k(x, x'). A LINEAR SVM on phi(x) is then a non-linear SVM on x.
    """
    def __init__(self, gamma: float = 0.1, n_components: int = 300, random_state: int = 42):
        self.gamma, self.D, self.random_state = gamma, n_components, random_state
 
    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        d = np.asarray(X).shape[1]
        self.W = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.D))
        self.b = rng.uniform(0.0, 2.0 * np.pi, self.D)
        return self
 
    def transform(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(np.asarray(X, dtype=float) @ self.W + self.b)
 
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
 
