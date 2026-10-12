"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.
Role: Data Engineer & EDA Specialist
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/bina_az_sale.csv"

# Columns that LEAK the target (derived from price). Drop these before training.
LEAKAGE_COLUMNS: list[str] = [
    "unit_price",
    "total_price",
]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError:
        try:
            return pd.read_csv(path, encoding="latin1")
        except Exception as e:
            raise FileNotFoundError(f"Could not load dataset from {path}. Error: {e}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers,
      - drop LEAKAGE_COLUMNS.
    """
    df = df.copy()

    # 1. Удаление дубликатов
    df = df.drop_duplicates()

    # 2. Удаление колонок с утечкой данных
    cols_to_drop = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 3. Универсальная очистка числовых колонок от текста, валют и пробелов
    # Проверяем наличие ключевых колонок из проекта (price, Sahə, Otaq sayı, Mərtəbə)
    potential_numeric = ["price", "Sahə", "Otaq sayı", "Mərtəbə", "Torpaq sahəsi", "lat", "lng"]
    for col in potential_numeric:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[^\d.]", "", regex=True)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Удаление строк без критических данных (цена и площадь)
    required_cols = [c for c in ["price", "Sahə"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)

    # 5. Фильтрация явных выбросов
    if "price" in df.columns:
        df = df[(df["price"] > 100) & (df["price"] < 50_000_000)]
    if "Sahə" in df.columns:
        df = df[(df["Sahə"] > 5) & (df["Sahə"] < 5000)]

    # 6. Заполнение пропусков в категориальных столбцах
    cat_cols = ["Kateqoriya", "Təmir", "Binanın növü", "location", "city", "İpoteka", "Çıxarış"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot),
      - assemble X (features) and y (target `price`).
    """
    df = df.copy()

    if "price" not in df.columns:
        raise ValueError("Target column 'price' not found in dataframe.")

    y = df["price"].values
    X_df = df.drop(columns=["price"])

    # One-hot encoding для категориальных признаков
    X_encoded = pd.get_dummies(X_df, drop_first=True, dtype=float)
    
    feature_names = list(X_encoded.columns)
    X = X_encoded.values

    return X, y, feature_names


def make_tier_label(y_price: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, float]:
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
    """
    y_price = np.asarray(y_price)
    if threshold is None:
        threshold = float(np.median(y_price))
    
    y_tier = (y_price > threshold).astype(int)
    return y_tier, threshold


def train_val_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    val_size: float = 0.15, 
    test_size: float = 0.15, 
    seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic split into train / validation / test."""
    np.random.seed(seed)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    test_count = int(n * test_size)
    val_count = int(n * val_size)

    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + val_count]
    train_idx = indices[test_count + val_count :]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]


def standardize(X_tr: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardize features using TRAIN statistics only."""
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std[std == 0.0] = 1.0

    scaled_tr = (X_tr - mean) / std
    scaled_others = tuple((arr - mean) / std for arr in others)

    return (scaled_tr, *scaled_others)
