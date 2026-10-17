"""
ui/app.py — a small demo UI for the bina.az project (Streamlit).

Run from the repo root:
    streamlit run ui/app.py

What it does
  - trains OUR decision tree and OUR Pegasos SVM (and the scikit-learn twins) on the
    project's own split, on train + validation rows only,
  - lets you describe a flat and shows the predicted price (Task A) and price tier (Task B),
  - shows which features matter most for our tree,
  - shows the tables and figures that `python -m src.run_all` wrote.

Data: the real bina.az CSV (data/bina_az_sale.csv) when it exists, otherwise the seeded
synthetic listings. Synthetic data is only a demo of the machinery, never a project result.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                       # data_prep.DATA_PATH is a relative path

from src import benchmark, data_prep, model_api, report_tables, run_all, synthetic  # noqa: E402

LANGS = {"en": "English", "az": "Azərbaycanca"}

TEXT = {
    "en": {
        "title": "🏠 bina.az: price and tier estimate",
        "data": "Data", "real": "Real bina.az data", "synthetic": "Demo data (synthetic)",
        "demo_warning": "Demo data: generated listings only check that the code works. They are not "
                        "project results. To use the real data, put `data/bina_az_sale.csv` in place "
                        "(see `data/README.md`).",
        "not_ready": "The data is not ready: {}",
        "trained_on": "Trained on {n:,} listings (train + validation).",
        "tuned": "Settings: chosen on the validation split.",
        "untuned": "Settings: defaults (run `python -m src.run_all` to tune them).",
        "tabs": ["Prediction", "Results", "About"],
        "describe": "Describe the property", "estimate": "Estimate price",
        "fill_form": "Fill in the form and press “Estimate price”.",
        "task_a": "Task A: price", "task_b": "Task B: price tier",
        "threshold": "Tier threshold: {t:,.0f} AZN (median of the training prices). "
                     "Above it: premium; at or below it: standard.",
        "proba": "premium probability", "margin": "SVM margin (distance from the boundary)",
        "importance": "What matters most to our tree", "importance_series": "importance",
        "no_results": "There are no results yet. Run in a terminal: ",
        "figures": "Figures",
        "loading": "Loading the data and training our tree and SVM…",
        "about": """
**Idea.** From bina.az listings we predict the price of a flat or house in two ways and compare
**our models, written from scratch in NumPy**, with scikit-learn, honestly.

- **Task A (regression):** predict the price (on log price). Model: our decision tree.
- **Task B (classification):** premium or standard, i.e. above or below the median of the training
  prices. Models: our SVM (Pegasos, sub-gradient descent on the hinge loss) and our tree in
  classification mode.

**Fair comparison.** Same rows, same features, matched hyperparameters. Settings are chosen on the
validation split only, the test split is used once, the tier threshold comes from training prices
only, and the leakage columns (`unit_price`, `total_price`) are removed.

**Reproduce everything:** `python -m src.run_all` (tables, figures, `results/results.json`).
""",
        "labels": {
            "Sahə": "Area, m²", "Otaq sayı": "Rooms", "floor": "Floor", "total_floors": "Floors in building",
            "Torpaq sahəsi": "Land area, sot", "lat": "Latitude", "lng": "Longitude",
            "area_m2": "Area, m²", "rooms": "Rooms", "repaired": "Repaired", "mortgage": "Mortgage",
            "Kateqoriya": "Category", "Təmir": "Repair", "Çıxarış": "Bill of sale (çıxarış)",
            "İpoteka": "Mortgage", "location": "District / location", "city": "City"},
    },
    "az": {
        "title": "🏠 bina.az: qiymət və seqment qiymətləndirməsi",
        "data": "Məlumat", "real": "Real bina.az məlumatı", "synthetic": "Demo məlumat (sintetik)",
        "demo_warning": "Demo məlumat: yaradılmış elanlar yalnız kodun işlədiyini yoxlayır. Bunlar "
                        "layihənin nəticələri deyil. Real məlumatdan istifadə etmək üçün "
                        "`data/bina_az_sale.csv` faylını yerləşdirin (bax: `data/README.md`).",
        "not_ready": "Məlumat hazır deyil: {}",
        "trained_on": "{n:,} elan üzərində öyrədilib (train + validation).",
        "tuned": "Parametrlər: validasiya hissəsində seçilib.",
        "untuned": "Parametrlər: standart dəyərlər (tənzimləmək üçün `python -m src.run_all` işə salın).",
        "tabs": ["Proqnoz", "Nəticələr", "Layihə haqqında"],
        "describe": "Əmlakı təsvir edin", "estimate": "Qiyməti qiymətləndir",
        "fill_form": "Formu doldurun və “Qiyməti qiymətləndir” düyməsini basın.",
        "task_a": "Tapşırıq A: qiymət", "task_b": "Tapşırıq B: qiymət seqmenti",
        "threshold": "Seqment həddi: {t:,.0f} AZN (təlim qiymətlərinin medianı). "
                     "Ondan yüksək: premium; bərabər və ya aşağı: standard.",
        "proba": "premium ehtimalı", "margin": "SVM marjini (sərhəddən məsafə)",
        "importance": "Ağacımız üçün ən vacib əlamətlər", "importance_series": "əhəmiyyət",
        "no_results": "Hələlik nəticə yoxdur. Terminalda işə salın: ",
        "figures": "Qrafiklər",
        "loading": "Məlumat yüklənir, ağac və SVM öyrədilir…",
        "about": """
**Ideya.** bina.az elanlarına əsasən mənzil və ya evin qiymətini iki üsulla proqnozlaşdırırıq və
**NumPy-da sıfırdan yazdığımız modelləri** scikit-learn ilə dürüst şəkildə müqayisə edirik.

- **Tapşırıq A (reqressiya):** qiyməti proqnozlaşdırmaq (qiymətin loqarifmi üzrə). Model: bizim qərar ağacı.
- **Tapşırıq B (təsnifat):** premium və ya standard, yəni təlim qiymətlərinin medianından yuxarı və ya
  aşağı. Modellər: bizim SVM (Pegasos, hinge itkisi üzrə subqradiyent enmə) və təsnifat rejimində ağacımız.

**Ədalətli müqayisə.** Eyni sətirlər, eyni əlamətlər, uyğunlaşdırılmış hiperparametrlər. Parametrlər
yalnız validasiya hissəsində seçilir, test hissəsi bir dəfə istifadə olunur, seqment həddi yalnız təlim
qiymətlərindən hesablanır, sızma sütunları (`unit_price`, `total_price`) silinib.

**Hər şeyi təkrar etmək üçün:** `python -m src.run_all` (cədvəllər, qrafiklər, `results/results.json`).
""",
        "labels": {
            "Sahə": "Sahə, m²", "Otaq sayı": "Otaq sayı", "floor": "Mərtəbə", "total_floors": "Binada mərtəbə sayı",
            "Torpaq sahəsi": "Torpaq sahəsi, sot", "lat": "Enlik", "lng": "Uzunluq",
            "area_m2": "Sahə, m²", "rooms": "Otaq sayı", "repaired": "Təmirli", "mortgage": "İpoteka",
            "Kateqoriya": "Kateqoriya", "Təmir": "Təmir", "Çıxarış": "Çıxarış",
            "İpoteka": "İpoteka", "location": "Rayon / məkan", "city": "Şəhər"},
    },
}
LANG = "en"


def t(key: str):
    """The text for the language chosen in the sidebar."""
    return TEXT[LANG][key]


def label(name: str) -> str:
    return TEXT[LANG]["labels"].get(name, name)


GEO = ("lat", "lng")


# --- Data and models (cached: they are built once per data source) -----------------------
def real_data_exists() -> bool:
    return os.path.exists(data_prep.DATA_PATH)


@st.cache_data(show_spinner=False)
def load_data(source: str):
    if source == "real":
        return benchmark.load_real_data(with_table=True)
    X, price, names = synthetic.make_listings()
    return X, price, names, None


def load_config(source: str):
    """Final settings chosen by the pipeline on the validation split, if it has been run."""
    path = "results/results.json" if source == "real" else "results/synthetic/results.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle).get("config")
        if saved:
            return run_all.merge_config(benchmark.CONFIG, saved), True
    return benchmark.CONFIG, False


@st.cache_resource(show_spinner=False)
def train(source: str):
    X, price, names, table = load_data(source)
    config, tuned = load_config(source)
    train_rows, val_rows, _ = run_all.split_rows(price)              # test rows are never used here
    fit = np.sort(np.concatenate([train_rows, val_rows]))
    threshold = float(np.median(price[train_rows]))                  # training prices only
    tier, _ = data_prep.make_tier_label(price, threshold)
    log_price = config["log_price"]

    y_reg = np.log(price[fit]) if log_price else price[fit]
    regressors = {s["name"]: s["make"](len(fit)).fit(X[fit], y_reg)
                  for s in benchmark.model_specs("regression", config)}

    mean, std = X[fit].mean(axis=0), X[fit].std(axis=0)
    std[std == 0.0] = 1.0                                            # same rule as data_prep.standardize
    classifiers = {}
    for spec in benchmark.model_specs("classification", config):
        X_fit = (X[fit] - mean) / std if spec["scale"] else X[fit]
        classifiers[spec["name"]] = (spec["make"](len(fit)).fit(X_fit, tier[fit]), spec["scale"])

    return {"regressors": regressors, "classifiers": classifiers, "mean": mean, "std": std,
            "threshold": threshold, "log_price": log_price, "config": config, "tuned": tuned,
            "n_fit": len(fit), "names": names}


# --- The input form ---------------------------------------------------------------------
def categorical_columns(names, table):
    if table is None:
        return []
    return [c for c in data_prep.CATEGORICAL_FEATURES
            if c in table.columns and any(n.startswith(c + "_") for n in names)]


def flat_form(names, X, table):
    """Draw the inputs and return {numeric feature: value} and {categorical column: choice}."""
    cats = categorical_columns(names, table)
    numeric = [n for n in names if not any(n.startswith(c + "_") for c in cats)]
    numbers, choices = {}, {}
    left, right = st.columns(2)
    for i, name in enumerate(numeric):
        column = left if i % 2 == 0 else right
        values = X[:, names.index(name)]
        shown_name = label(name)
        if set(np.unique(values)) <= {0.0, 1.0}:
            numbers[name] = float(column.checkbox(shown_name, value=values.mean() > 0.5, key=name))
        elif name in GEO:
            known = values[values != 0.0]                            # 0 means "missing" after cleaning
            numbers[name] = column.number_input(shown_name, value=float(np.median(known)),
                                                step=0.001, format="%.4f", key=name)
        else:
            numbers[name] = column.number_input(shown_name, min_value=0.0, value=float(np.median(values)),
                                                step=1.0, key=name)
    for i, col in enumerate(cats):
        options = table[col].value_counts().index.tolist()           # most common first
        column = left if i % 2 == 0 else right
        def shown(value, col=col):                                   # col bound now, not at call time
            if LANG == "en" and col == "Kateqoriya" and value in report_tables.CATEGORY_NAMES:
                return f"{value} — {report_tables.CATEGORY_NAMES[value]}"
            return value
        choices[col] = column.selectbox(label(col), options, format_func=shown, key=col)
    return numbers, choices, cats


def feature_vector(names, numbers, choices) -> np.ndarray:
    """One row in the exact column order the models were trained on (dummies as get_dummies made them)."""
    row = np.zeros(len(names))
    for j, name in enumerate(names):
        if name in numbers:
            row[j] = numbers[name]
        else:
            for col, choice in choices.items():
                if name == f"{col}_{choice}":
                    row[j] = 1.0
    return row.reshape(1, -1)


# --- Pages ------------------------------------------------------------------------------
def page_predict(source: str, models: dict) -> None:
    X, _, names, table = load_data(source)
    st.subheader(t("describe"))
    with st.form("flat"):
        numbers, choices, _ = flat_form(names, X, table)
        submitted = st.form_submit_button(t("estimate"), type="primary")

    x = feature_vector(names, numbers, choices)
    if not submitted and "shown_once" not in st.session_state:
        st.info(t("fill_form"))
        return
    st.session_state["shown_once"] = True

    st.subheader(t("task_a"))
    columns = st.columns(len(models["regressors"]))
    for column, (name, model) in zip(columns, models["regressors"].items()):
        raw = float(model.predict(x)[0])
        price = float(np.exp(raw)) if models["log_price"] else raw
        column.metric(name, f"{price:,.0f} AZN")

    st.subheader(t("task_b"))
    st.caption(t("threshold").format(t=models["threshold"]))
    columns = st.columns(len(models["classifiers"]))
    for column, (name, (model, scaled)) in zip(columns, models["classifiers"].items()):
        x_in = (x - models["mean"]) / models["std"] if scaled else x
        premium = int(model.predict(x_in)[0]) == 1
        score = benchmark.ranking_scores(model, x_in)
        column.metric(name, "premium" if premium else "standard")
        if score is not None:
            kind = t("proba") if hasattr(model, "predict_proba") else t("margin")
            column.caption(f"{kind}: {float(score[0]):.3f}")

    tree = models["regressors"].get("Ours: tree")
    importances = model_api.feature_importances(tree) if tree is not None else None
    if importances is not None and importances.sum() > 0:
        st.subheader(t("importance"))
        top = np.argsort(-importances)[:8]
        st.bar_chart({t("importance_series"): {names[i]: float(importances[i]) for i in top}}, horizontal=True)


def page_results(source: str) -> None:
    folder = "results" if source == "real" else os.path.join("results", "synthetic")
    summary = os.path.join(folder, "summary.md")
    if not os.path.exists(summary):
        st.warning(t("no_results")
                   + ("`python -m src.run_all`" if source == "real" else "`python -m src.run_all --synthetic`"))
        return
    with open(summary, encoding="utf-8") as handle:
        st.markdown(handle.read())
    fig_dir = os.path.join(folder, "figures") if source != "real" else os.path.join("report", "figures")
    pictures = sorted(f for f in os.listdir(fig_dir) if f.endswith(".png")) if os.path.isdir(fig_dir) else []
    if pictures:
        st.subheader(t("figures"))
        columns = st.columns(2)
        for i, name in enumerate(pictures):
            columns[i % 2].image(os.path.join(fig_dir, name), caption=name, width="stretch")


def page_about() -> None:
    st.markdown(t("about"))


# --- App --------------------------------------------------------------------------------
def main() -> None:
    global LANG
    st.set_page_config(page_title="bina.az", page_icon="🏠", layout="wide")
    LANG = st.sidebar.radio("Language / Dil", list(LANGS), format_func=LANGS.get, key="lang")
    st.title(t("title"))

    options = (["real"] if real_data_exists() else []) + ["synthetic"]
    source = st.sidebar.radio(t("data"), options, format_func=lambda key: t(key))
    if source == "synthetic":
        st.warning(t("demo_warning"))

    try:
        with st.spinner(t("loading")):
            models = train(source)
    except SystemExit as stop:                                      # data stage guards in load_real_data
        st.error(t("not_ready").format(stop))
        st.stop()

    st.sidebar.caption(t("trained_on").format(n=models["n_fit"]))
    st.sidebar.caption(t("tuned") if models["tuned"] else t("untuned"))

    tab_predict, tab_results, tab_about = st.tabs(t("tabs"))
    with tab_predict:
        page_predict(source, models)
    with tab_results:
        page_results(source)
    with tab_about:
        page_about()


main()
