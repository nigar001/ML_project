# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Karimova Ulkar         | @iamulkeer    | Data pipeline design, cleaning, leakage removal, feature engineering & EDA |
| Dilavarli Farid        | @faridqul     | Evaluation metrics, benchmark against scikit-learn, one-command pipeline, report and slides |
| ---                    | ---           | ---                   |

## Per-member summary
**Karimova Ulkar —** Designed and implemented the comprehensive data preprocessing and cleaning pipeline in `src/data_prep.py`. Handled type coercion, currency parsing, missing value imputation, duplicate removal, and outlier filtering. Removed target leakage columns (`unit_price`, `total_price`), derived the median-based training price tier label, and built structured train/val/test splits along with feature scaling functions. Conducted EDA and generated core distribution plots.

**Dilavarli Farid —** Wrote the evaluation metrics in NumPy in `src/evaluate.py` (RMSE, MAE, R², confusion matrix, precision/recall/F1, rank-based ROC-AUC, ROC curve points, the comparison table) with tests in `tests/test_evaluate.py` that check them against hand-worked examples and scikit-learn. Built the benchmark in `src/benchmark.py`: scikit-learn baselines with matched settings, seeded stratified 5-fold cross-validation reporting mean ± std, fit/predict timing, and agreement checks between our models and the library's, with tests in `tests/test_benchmark.py`. Profiled the raw dataset and reported the repeat-scrape leak and parsing problems that the data stage then fixed. Used Claude Code as an assistant for this work (see the co-author lines in the commits). Still to come: `src/run_all.py`, the report and the slides.

**Surname, Name —** ...

## Rough effort split
Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member | Approx. % |
|--------|-----------|
| Ulkar  | 25%       |
| ---    | ---       |
| ---    | ---       |
