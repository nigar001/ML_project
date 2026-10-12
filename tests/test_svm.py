"""SVM-only tests.   Run:  python -m tests.test_svm"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from src.svm import PegasosSVM, RandomFourierFeatures
from src import svm_experiments as ex


def blobs(n=400, sep=2.0, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-sep, noise, (n // 2, 2)), rng.normal(sep, noise, (n // 2, 2))])
    return X, np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)


def objective(w, b, X, y, lam):
    ypm = np.where(y == 1, 1.0, -1.0)
    return 0.5 * lam * (w @ w) + np.mean(np.maximum(0, 1 - ypm * (X @ w + b)))


def test_objective_decreases_and_classifies():
    X, y = blobs()
    m = PegasosSVM(lambda_=1e-3, n_iters=3000).fit(X, y)
    assert m.history_[-1][1] < 0.5 * m.history_[0][1] and m.score(X, y) > 0.95


def test_labels_are_mapped_back():
    X, y = blobs()
    m = PegasosSVM(lambda_=1e-3, n_iters=500).fit(X, np.where(y == 1, 7, 3))
    assert set(np.unique(m.predict(X))) <= {3, 7} and m.classes_.tolist() == [3, 7]


def test_margin_and_support_vectors_definition():
    X, y = blobs(sep=1.0)
    m = PegasosSVM(lambda_=1e-2, n_iters=3000).fit(X, y)
    ypm = np.where(y == 1, 1.0, -1.0)
    assert np.isclose(m.margin_width_, 2 / np.linalg.norm(m.w))
    sv = set(np.where(ypm * (X @ m.w + m.b) <= 1.0)[0])
    assert sv == set(m.support_) and 0 < len(sv) <= len(y)


def test_larger_lambda_gives_wider_margin():
    X, y = blobs(sep=1.0)
    widths = [PegasosSVM(lambda_=l, n_iters=4000).fit(X, y).margin_width_ for l in (1e-4, 1e-3, 1e-2, 1e-1)]
    assert all(a < b for a, b in zip(widths, widths[1:])), widths


def test_n_epochs_overrides_n_iters_and_batch_size_one_works():
    X, y = blobs(n=200)
    m = PegasosSVM(lambda_=1e-2, n_epochs=3, batch_size=10, n_iters=1).fit(X, y)
    assert m.n_steps_ == 60 and np.isclose(m.epochs_, 3.0)
    assert PegasosSVM(lambda_=1e-2, n_iters=2000, batch_size=1).fit(X, y).score(X, y) > 0.9


def test_schedules_all_converge():
    X, y = blobs()
    for kw in (dict(schedule="pegasos"), dict(schedule="sqrt", eta0=0.1), dict(schedule="constant", eta0=0.01)):
        m = PegasosSVM(lambda_=1e-2, n_iters=3000, **kw).fit(X, y)
        assert m.score(X, y) > 0.95, kw


def test_reproducible_with_same_seed():
    X, y = blobs()
    a = PegasosSVM(lambda_=1e-3, n_iters=500, random_state=1).fit(X, y)
    b = PegasosSVM(lambda_=1e-3, n_iters=500, random_state=1).fit(X, y)
    assert np.array_equal(a.w, b.w) and a.b == b.b


def test_nan_input_raises_clear_error():
    X, y = blobs(); X[0, 0] = np.nan
    try:
        PegasosSVM().fit(X, y)
    except ValueError as e:
        assert "NaN" in str(e)
    else:
        raise AssertionError("NaN should be rejected")


def test_matches_sklearn_linearsvc_objective_and_ranking():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 6)); y = (X @ rng.normal(size=6) + rng.normal(0, 1.0, 800) > 0).astype(int)
    lam = 1e-2; n = len(y)
    ours = PegasosSVM(lambda_=lam, n_epochs=60, batch_size=16).fit(X, y)
    sk = LinearSVC(C=1 / (n * lam), loss="hinge", dual=True, max_iter=100000).fit(X, y)
    J_ours = objective(ours.w, ours.b, X, y, lam); J_sk = objective(sk.coef_[0], sk.intercept_[0], X, y, lam)
    assert J_ours < J_sk * 1.05, (J_ours, J_sk)                           # within 5% of the sklearn optimum
    assert abs(roc_auc_score(y, ours.decision_function(X)) - roc_auc_score(y, sk.decision_function(X))) < 0.01
    assert (ours.predict(X) == sk.predict(X)).mean() > 0.97


def test_random_fourier_features_solve_nonlinear_problem():
    rng = np.random.default_rng(4); X = rng.normal(size=(800, 2)); y = (np.linalg.norm(X, axis=1) > 1.2).astype(int)
    lin = PegasosSVM(lambda_=1e-4, n_iters=4000).fit(X, y)
    rff = RandomFourierFeatures(gamma=1.0, n_components=300).fit(X)
    non = PegasosSVM(lambda_=1e-4, n_iters=4000).fit(rff.transform(X), y)
    assert non.score(rff.transform(X), y) > lin.score(X, y) + 0.15


def test_rff_approximates_rbf_kernel():
    rng = np.random.default_rng(5); X = rng.normal(size=(40, 3)); gamma = 0.3
    Z = RandomFourierFeatures(gamma, 8000).fit(X).transform(X)
    K = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(-1))
    assert np.abs(Z @ Z.T - K).max() < 0.06


def test_metrics_in_experiments_match_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(6); y = rng.integers(0, 2, 400); s = rng.normal(size=400) + y; s[:40] = np.round(s[:40])
    assert np.isclose(ex.auc(y, s), roc_auc_score(y, s)) and np.isclose(ex.f1(y, s > .5), f1_score(y, s > .5))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)