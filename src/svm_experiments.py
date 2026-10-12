"""
svm_experiments.py — Member 3 (SVM). Every study for Problem 3 (b, c, d), Problem 4 (b) and the SVM
rows of Problem 5.

    python -m src.svm_experiments

Depends ONLY on the team's data_prep API, unchanged:
    load_raw, clean, make_features, make_tier_label, train_val_test_split, standardize
and adds a small "SVM-ready" layer on top (prepare / load_tier_data) so that identifier columns,
free text and NaNs cannot break the SVM. Hyperparameters are chosen on VALIDATION only; the test set is
evaluated exactly once at the very end with the frozen configs.

Outputs: console tables, report/figures/svm_*.png, report/svm_results.json
"""
from __future__ import annotations

import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data_prep as dp
from .svm import PegasosSVM, RandomFourierFeatures

FIG = "report/figures"
SEED = getattr(dp, "SEED", 42)
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
EPOCHS = 20            # passes over the training data in the sweeps
BATCH = 32
MAX_LEVELS = 30        # categorical columns keep at most this many levels (others -> "other")
ID_LIKE = {"id", "url", "link", "phone", "owner name", "shop name", "address", "description",
           "attributes", "unit price", "total price"}
LEAK_CORR = 0.95       # a feature this correlated with price / log(price) is treated as leakage


def _fold(name) -> str:
    return str(name).lower().replace("_", " ").strip()


# =============================================================== SVM-ready data layer
def _train_rows(n: int, price: np.ndarray) -> np.ndarray:
    """Row numbers of the TRAIN split, obtained by pushing an index column through the team's own split."""
    idx = np.arange(n, dtype=float).reshape(-1, 1)
    return dp.train_val_test_split(idx, price)[0][:, 0].astype(int)


def prepare(df: pd.DataFrame):
    """
    Cleaned frame -> frame that is safe for one-hot encoding:
      * drop identifier / free-text / leakage-named columns (id, owner_name, address, description, ...);
      * cap every text column at MAX_LEVELS levels. The kept levels are chosen from TRAIN rows only,
        so validation/test rows never influence the encoding.
    """
    df = df.copy()
    drop = [c for c in df.columns if c != "price" and _fold(c) in ID_LIKE]
    df = df.drop(columns=drop)
    tr = _train_rows(len(df), df["price"].to_numpy(float))
    for c in df.columns:
        if c == "price" or pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c].astype(str)
        keep = s.iloc[tr].value_counts().index[:MAX_LEVELS]
        df[c] = s.where(s.isin(keep), "other")
    return df, drop, tr


def load_tier_data():
    """Run the team pipeline and return everything the experiments need (Task B: price tier)."""
    raw = dp.load_raw()
    df = dp.clean(raw)
    if len(df) == 0:
        raise RuntimeError(
            "dp.clean() returned 0 rows. With pandas >= 3 its `dtype == object` test never matches string "
            "columns, so all numbers become NaN. Install the pinned version:  pip install pandas==2.2.2")
    df, dropped, tr_rows = prepare(df)
    X, price, names = dp.make_features(df)
    X, price, names = np.asarray(X, dtype=float), np.asarray(price, dtype=float), list(names)

    # automatic leakage guard (fitted on TRAIN rows): drop features that (almost) encode the price
    ly = np.log(price[tr_rows])
    bad = []
    for j in range(X.shape[1]):
        col = X[tr_rows, j]
        if np.nanstd(col) < 1e-12 or np.isnan(col).any():
            continue
        r = max(abs(np.corrcoef(col, price[tr_rows])[0, 1]), abs(np.corrcoef(col, ly)[0, 1]))
        if r > LEAK_CORR:
            bad.append((names[j], round(float(r), 3)))
    if bad:
        print("!! dropping features that look like target leakage:", bad)
        keep_cols = [j for j, n in enumerate(names) if n not in {b[0] for b in bad}]
        X, names = X[:, keep_cols], [names[j] for j in keep_cols]

    Xtr, ptr, Xva, pva, Xte, pte = dp.train_val_test_split(X, price)
    ytr, thr = dp.make_tier_label(ptr)                       # threshold = TRAIN median only
    yva, _ = dp.make_tier_label(pva, thr)
    yte, _ = dp.make_tier_label(pte, thr)

    # impute remaining NaN (e.g. missing lat/lng) with TRAIN medians
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr, Xva, Xte = (np.where(np.isnan(A), med, A) for A in (Xtr, Xva, Xte))
    Str, Sva, Ste = dp.standardize(Xtr, Xva, Xte)             # TRAIN mean/std only

    print(f"raw {len(raw)} rows -> clean {len(df)} rows; dropped identifier columns: {dropped}")
    print(f"features {X.shape[1]} | split train/val/test = {len(ytr)}/{len(yva)}/{len(yte)} | "
          f"tier threshold (train median) = {thr:,.0f} | premium share train/val/test = "
          f"{ytr.mean():.3f}/{yva.mean():.3f}/{yte.mean():.3f}")
    return dict(Xtr=Xtr, Xva=Xva, Xte=Xte, Str=Str, Sva=Sva, Ste=Ste, ytr=ytr, yva=yva, yte=yte,
                names=names, threshold=thr, pte=pte)


# =============================================================== metrics (NumPy, no sklearn)
def confusion(y, p):
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tp = int(((y == 1) & (p == 1)).sum())
    return [[tn, fp], [fn, tp]]


def prf(y, p):
    (tn, fp), (fn, tp) = confusion(y, p)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def f1(y, p) -> float:
    return prf(y, p)[2]


def auc(y, s) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_svm(lam, S, y, **kw):
    kw.setdefault("n_epochs", EPOCHS)
    kw.setdefault("batch_size", BATCH)
    kw.setdefault("record_every", 200)
    return PegasosSVM(lambda_=lam, random_state=SEED, **kw).fit(S, y)


def cv_scores(fit_predict, X, y, k=5):
    """k-fold CV on train+val (test never touched). fit_predict(Xa, ya, Xb) -> (labels, scores)."""
    folds = np.array_split(np.random.default_rng(SEED).permutation(len(y)), k)
    f, a = [], []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        pred, score = fit_predict(X[tr], y[tr], X[te])
        f.append(f1(y[te], pred)); a.append(auc(y[te], score))
    return (float(np.mean(f)), float(np.std(f))), (float(np.mean(a)), float(np.std(a)))


# =============================================================== experiments
def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    D = load_tier_data()
    Str, Sva, ytr, yva, names = D["Str"], D["Sva"], D["ytr"], D["yva"], D["names"]
    d = Str.shape[1]
    out: dict = {"n_features": d, "tier_threshold": D["threshold"]}

    # ---- (b) lambda sweep: margin vs validation metric --------------------------------------
    rows = []
    for lam in LAMBDAS:
        m = fit_svm(lam, Str, ytr)
        rows.append(dict(lam=lam, w_norm=m.w_norm_, margin=m.margin_width_, n_sv=int(len(m.support_)),
                         train_obj=m.history_[-1][1], train_f1=f1(ytr, m.predict(Str)),
                         val_f1=f1(yva, m.predict(Sva))))
    print("\n[Problem 4b] lambda sweep (linear Pegasos, %d epochs, batch %d)" % (EPOCHS, BATCH))
    print(" lambda     ||w||   margin   #SV   train_obj  train_F1   val_F1")
    for r in rows:
        print(f" {r['lam']:.0e}  {r['w_norm']:8.3f} {r['margin']:8.3f} {r['n_sv']:6d}  {r['train_obj']:9.4f}  "
              f"{r['train_f1']:8.4f}  {r['val_f1']:7.4f}")
    best_lam = max(rows, key=lambda r: r["val_f1"])["lam"]
    out.update(lambda_sweep=rows, best_lambda=best_lam)
    print(f"-> best lambda on VALIDATION: {best_lam:.0e}")

    L = [r["lam"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].semilogx(L, [r["train_f1"] for r in rows], "o-", color="#2a6fbb", label="train F1")
    ax[0].semilogx(L, [r["val_f1"] for r in rows], "s--", color="#c8641e", label="validation F1")
    ax[0].set(xlabel="lambda (regularisation)", ylabel="F1 (premium)", title="Regularisation vs F1"); ax[0].legend()
    ax[1].loglog(L, [r["margin"] for r in rows], "o-", color="#2a6fbb")
    ax[1].set(xlabel="lambda (regularisation)", ylabel="margin width 2/||w||", title="Stronger lambda = wider margin")
    for a in ax:
        a.grid(True, color="#d9d9d9", linewidth=0.6); a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIG}/svm_lambda_sweep.png", dpi=200); plt.close()

    # ---- (a) objective convergence + learning-rate schedules ---------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sched = {}
    for name, kw, c in [("Pegasos 1/(lambda t)", dict(schedule="pegasos"), "#2a6fbb"),
                        ("1/sqrt(t), eta0=0.1", dict(schedule="sqrt", eta0=0.1), "#c8641e"),
                        ("constant, eta0=0.01", dict(schedule="constant", eta0=0.01), "#6b6b6b")]:
        m = fit_svm(best_lam, Str, ytr, **kw)
        t, v = zip(*m.history_)
        ax.plot(t, v, color=c, label=name)
        sched[name] = dict(final_obj=float(m.history_[-1][1]), val_f1=f1(yva, m.predict(Sva)))
    ax.set(yscale="log", xlabel="iteration", ylabel="objective J(w,b)", title="Training objective converges")
    ax.legend(fontsize=8); ax.grid(True, color="#d9d9d9", linewidth=0.6); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIG}/svm_convergence.png", dpi=200); plt.close()
    out["schedules"] = sched
    print("\n[Problem 3a/3b] learning-rate schedules at best lambda (final objective, val F1):")
    for k, v in sched.items():
        print(f"  {k:22s} J={v['final_obj']:.4f}  val F1={v['val_f1']:.4f}")

    # ---- epochs and batch size ----------------------------------------------------------------
    ep = {}
    for e in (1, 2, 5, 20, 50):
        m = fit_svm(best_lam, Str, ytr, n_epochs=e)
        ep[e] = dict(final_obj=float(m.history_[-1][1]), val_f1=f1(yva, m.predict(Sva)))
    bs = {}
    for b in (1, 32, 256):
        t0 = time.perf_counter(); m = fit_svm(best_lam, Str, ytr, batch_size=b, n_epochs=5)
        bs[b] = dict(val_f1=f1(yva, m.predict(Sva)), fit_s=time.perf_counter() - t0)
    out.update(epochs=ep, batch_size=bs)
    print("\n[Problem 3b] epochs ->", {k: (round(v['final_obj'], 4), round(v['val_f1'], 4)) for k, v in ep.items()},
          "(final objective, val F1)")
    print("             batch size (5 epochs) ->", {k: (round(v['val_f1'], 4), round(v['fit_s'], 2)) for k, v in bs.items()},
          "(val F1, seconds)")

    # ---- (c) margin and support vectors of the chosen linear model ------------------------------
    lin = fit_svm(best_lam, Str, ytr)
    top = np.argsort(-np.abs(lin.w))[:10]
    print(f"\n[Problem 3c] linear SVM, lambda={best_lam:.0e}: ||w||={lin.w_norm_:.3f}, margin width 2/||w||="
          f"{lin.margin_width_:.3f}, support vectors={len(lin.support_)}/{len(ytr)} "
          f"({100 * len(lin.support_) / len(ytr):.1f}%)")
    print("  top |weights| (standardised features):", [(names[i], round(float(lin.w[i]), 3)) for i in top])
    out["linear"] = dict(w_norm=lin.w_norm_, margin=lin.margin_width_, n_sv=int(len(lin.support_)),
                         val_f1=f1(yva, lin.predict(Sva)), top_weights=[(names[i], float(lin.w[i])) for i in top])
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.hist(lin.train_margins_, bins=60, range=(-4, 6), color="#2a6fbb")
    ax.axvline(1, color="#c8641e", linestyle="--", label="y(w.x+b) = 1")
    ax.set(xlabel="functional margin y(w.x+b)", ylabel="training points",
           title="Points left of the line are support vectors"); ax.title.set_fontsize(9); ax.legend()
    ax.grid(True, color="#d9d9d9", linewidth=0.6); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIG}/svm_margins.png", dpi=200); plt.close()

    # ---- (d) non-linearity: random Fourier features, gamma x lambda ------------------------------
    D_RFF = 500
    res = []
    for gm in (0.02, 0.1, 0.5, 2.0):      # if the best value sits at the EDGE of this grid, extend the grid
        gamma = gm / d                                   # gamma scaled by 1/d: a standard RBF default
        rff = RandomFourierFeatures(gamma, D_RFF, SEED).fit(Str)
        Ftr, Fva = rff.transform(Str), rff.transform(Sva)
        for lam in (1e-4, 1e-3, 1e-2):
            m = fit_svm(lam, Ftr, ytr, record_every=2000)
            res.append(dict(gamma_mult=gm, gamma=gamma, lam=lam, D=D_RFF, margin=m.margin_width_,
                            val_f1=f1(yva, m.predict(Fva))))
    best_rff = max(res, key=lambda r: r["val_f1"])
    out.update(rff_sweep=res, best_rff=best_rff)
    print(f"\n[Problem 3d / 4b] RFF sweep (D={D_RFF}); gamma = mult/d, d={d}")
    print("  mult  lambda   val F1  margin")
    for r in res:
        print(f"  {r['gamma_mult']:<5} {r['lam']:.0e}  {r['val_f1']:.4f}  {r['margin']:.3f}")
    print(f"-> best RFF on VALIDATION: gamma={best_rff['gamma']:.4f} lambda={best_rff['lam']:.0e} "
          f"(val F1 {best_rff['val_f1']:.4f}) vs linear (val F1 {out['linear']['val_f1']:.4f})")

    # ---- final configs are now FROZEN. Cross-validation (train+val) ---------------------------------
    Xtv = np.vstack([D["Xtr"], D["Xva"]]); ytv = np.r_[ytr, yva]

    def cv_linear(Xa, ya, Xb):
        sa, sb = dp.standardize(Xa, Xb)
        m = fit_svm(best_lam, sa, ya, record_every=10_000)
        return m.predict(sb), m.decision_function(sb)

    def cv_rff(Xa, ya, Xb):
        sa, sb = dp.standardize(Xa, Xb)
        rf = RandomFourierFeatures(best_rff["gamma"], D_RFF, SEED).fit(sa)
        m = fit_svm(best_rff["lam"], rf.transform(sa), ya, record_every=10_000)
        return m.predict(rf.transform(sb)), m.decision_function(rf.transform(sb))

    from sklearn.linear_model import SGDClassifier            # baselines only (Problem 5b)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    def sgd_pipe():
        return make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=best_lam, max_iter=50, tol=None,
                                                             random_state=SEED))

    def cv_sgd(Xa, ya, Xb):
        p = sgd_pipe().fit(Xa, ya)
        return p.predict(Xb), p.decision_function(Xb)

    cv = {"MyPegasos-linear": cv_scores(cv_linear, Xtv, ytv), "MyPegasos-RFF": cv_scores(cv_rff, Xtv, ytv),
          "sk-SGD-hinge": cv_scores(cv_sgd, Xtv, ytv)}
    out["cv_5fold"] = cv
    print("\n[Problem 5c] 5-fold CV on train+val, mean +- std  (F1 | ROC-AUC)")
    for k, ((fm, fs), (am, as_)) in cv.items():
        print(f"  {k:18s} F1 {fm:.4f} +- {fs:.4f} | AUC {am:.4f} +- {as_:.4f}")

    # ---- single, final TEST evaluation (configs frozen above) -------------------------------------------
    Ste = D["Ste"]; yte = D["yte"]
    rows_t = {}

    t0 = time.perf_counter(); m_lin = fit_svm(best_lam, Str, ytr); t_lin = time.perf_counter() - t0
    rows_t["MyPegasos-linear"] = (m_lin.predict(Ste), m_lin.decision_function(Ste), t_lin)

    rff = RandomFourierFeatures(best_rff["gamma"], D_RFF, SEED).fit(Str)
    t0 = time.perf_counter(); m_rff = fit_svm(best_rff["lam"], rff.transform(Str), ytr); t_rff = time.perf_counter() - t0
    rows_t["MyPegasos-RFF"] = (m_rff.predict(rff.transform(Ste)), m_rff.decision_function(rff.transform(Ste)), t_rff)

    t0 = time.perf_counter(); p = sgd_pipe().fit(D["Xtr"], ytr); t_sgd = time.perf_counter() - t0
    rows_t["sk-SGD-hinge"] = (p.predict(D["Xte"]), p.decision_function(D["Xte"]), t_sgd)

    n = len(ytr)
    t0 = time.perf_counter()
    lsvc = make_pipeline(StandardScaler(), LinearSVC(C=1.0 / (n * best_lam), loss="hinge", dual=True,
                                                     max_iter=20_000, random_state=SEED)).fit(D["Xtr"], ytr)
    t_lsvc = time.perf_counter() - t0
    rows_t["sk-LinearSVC"] = (lsvc.predict(D["Xte"]), lsvc.decision_function(D["Xte"]), t_lsvc)

    print("\n[Problem 5 / 6] TEST metrics (configs chosen on validation; test evaluated once)")
    print("  model               acc    prec   recall     F1    ROC-AUC  fit_s")
    final = {}
    for k, (pred, score, secs) in rows_t.items():
        pr, rc, f = prf(yte, pred)
        final[k] = dict(acc=float(np.mean(pred == yte)), precision=pr, recall=rc, f1=f, roc_auc=auc(yte, score),
                        fit_s=secs, confusion=confusion(yte, pred))
        v = final[k]
        print(f"  {k:18s} {v['acc']:.4f} {pr:.4f}  {rc:.4f}  {f:.4f}  {v['roc_auc']:.4f}  {secs:.2f}")
    print("  confusion [[TN, FP], [FN, TP]] MyPegasos-linear:", final["MyPegasos-linear"]["confusion"])
    out["test"] = final

    # ---- error analysis (Problem 6): where does the linear SVM fail? (descriptive, nothing is tuned here) ----
    pte, thr = D["pte"], D["threshold"]
    wrong = rows_t["MyPegasos-linear"][0] != yte
    rel = np.abs(pte / thr - 1.0)                       # relative distance of the true price from the tier threshold
    print("\n[Problem 6] linear SVM error rate by distance of the true price from the tier threshold:")
    ea = {}
    for lo, hi in ((0, .10), (.10, .25), (.25, .50), (.50, np.inf)):
        mk = (rel >= lo) & (rel < hi)
        if mk.any():
            ea[f"{lo:.2f}-{hi:.2f}"] = dict(n=int(mk.sum()), error_rate=float(wrong[mk].mean()))
            print(f"  within {lo:>4.0%}-{hi:>4.0%} of threshold: n={int(mk.sum()):5d}  error rate {wrong[mk].mean():.3f}")
    top = pte >= np.quantile(pte, 0.99)
    ea["top_1pct_price"] = dict(n=int(top.sum()), error_rate=float(wrong[top].mean()))
    print(f"  most expensive 1% listings: n={int(top.sum())}  error rate {wrong[top].mean():.3f}")
    out["error_analysis"] = ea
    json.dump(out, open("report/svm_results.json", "w"), indent=2, default=float)
    print("\nSaved report/svm_results.json and report/figures/svm_*.png")


if __name__ == "__main__":
    main()