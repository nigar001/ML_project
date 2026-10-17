# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team

| Member (Surname, Name) | GitHub handle | Main responsibilities                                                                          |
| ---------------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| Karimova Ulkar         | @iamulkeer    | Data pipeline design, cleaning, leakage removal, feature engineering & EDA                     |
| Dilavarli Farid        | @faridqul     | Evaluation metrics, benchmark, one-command pipeline (tuning, error analysis), report, slides   |
| Alimammadova Nigar     | @nigar001     | Pegasos SVM (hinge-loss, random Fourier features), SVM experiments and tests, repository setup |
| ---                    | ---           | ---                                                                                            |

## Per-member summary

**Karimova Ulkar —** Designed and implemented the data pipeline in src/data_prep.py and the EDA script src/eda_plots.py. Loaded the bina.az CSV (UTF-8 with a latin1 fallback) and deduplicated listings by URL. Parsed prices, floors ("7 / 9" into floor and total_floors) and areas (converted sot to m²). Filtered outliers (price 1,000-10,000,000 AZN, area 15-3,000 m², up to 50,000 m² for land plots) and imputed missing values with constants, so no test statistics leak into training. Removed the target-leakage columns unit_price and total_price, built the feature matrix from explicit numeric and categorical whitelists with one-hot encoding, and implemented the median-based premium/standard price tier label, with an optional fixed threshold so that the training median can be reused for validation and test. Implemented a seeded 70/15/15 train/val/test split with optional stratification and train-only standardization. In eda_plots.py, generated five figures: price and log-price distributions, area vs. price, rooms distribution, a lat/lng price map and tier class balance.

**Dilavarli Farid —** Wrote the evaluation metrics in NumPy in `src/evaluate.py` (RMSE, MAE, R², confusion matrix, precision/recall/F1, rank-based ROC-AUC with ties, ROC curve points, the comparison table), tested against hand-worked examples and scikit-learn. Built the benchmark in `src/benchmark.py`: scikit-learn baselines with matched settings (trees with the same stopping rules, `SGDClassifier` with hinge loss and α = λ inside a scaling `Pipeline`), seeded stratified 5-fold cross-validation reporting mean ± std, fit/predict timing, and agreement checks between our models and the library's. Built the one-command pipeline `src/run_all.py`: one stratified train/validation/test split, the tuning stage (`src/tuning.py`: λ sweep for the SVM, depth / criterion / leaf-size sweeps for the tree, on validation data only), one final evaluation on the test split, diagnostics (confusion matrices, ROC curves, feature importances, SVM weights, margin, support vectors), the error analysis by price band, property type and district size (`src/error_analysis.py`), the figures (`src/report_figures.py`) and the report's LaTeX tables (`src/report_tables.py`); `src/model_api.py` adapts the pipeline to the team's tree and SVM. Wrote the tests for all of these (`tests/test_evaluate.py`, `test_benchmark.py`, `test_run_all.py`, `test_tuning.py`, `test_error_analysis.py`, `test_report_tables.py`). Profiled the raw dataset and reported the repeat-scrape leak and parsing problems that the data stage then fixed. Wrote the report (`report/report.tex`) and the slides with the team's input. Used Claude Code as a coding and writing assistant for this work (see the co-author lines in the commits and the report's AI-assistance statement).

**Alimammadova Nigar —** Implemented the linear soft-margin SVM in `src/svm.py`: Pegasos mini-batch sub-gradient descent on the hinge-loss primal (batch size 32, η = 1/(λt), bias folded into the weights, optional projection), which records the objective history, the margin width 2/‖w‖ and the support vectors, plus random Fourier features as the non-linear extension. Wrote `src/svm_experiments.py` (λ sweep, learning-rate schedules, epochs and batch size, margin and support-vector analysis, random-Fourier-feature sweep, 5-fold cross-validation against scikit-learn's SGDClassifier and LinearSVC, error analysis by distance from the tier threshold) and `tests/test_svm.py` (12 tests, e.g. objective decreases, larger λ gives a wider margin, final objective close to LinearSVC, RFF approximates the RBF kernel). Chose λ = 0.001 on the validation split, ran the experiments on the real data, and set up and pushed to the shared repository. Used Claude as an AI assistant for this work, to draft the code, experiments and tests and to explain the method.

**Surname, Name —** ...

## Rough effort split

Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member | Approx. % |
| ------ | --------- |
| Ulkar  | 25%       |
| Nigar  | 25%       |
| Farid  | 25%       |
| Nihad  | 25%       |
