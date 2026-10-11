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
    # Добавьте сюда колонки, если они есть в вашем датасете и зависят от цены (например, цена за м2)
    "unit_price",
    "total_price",
]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please download 'binaaz-sale-project' from Kaggle "
            "and place the CSV into the data/ folder."
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers (document your rule),
      - drop LEAKAGE_COLUMNS.
    Return the cleaned DataFrame.
    """
    df = df.copy()

    # 1. Удаление дубликатов
    df = df.drop_duplicates()

    # 2. Удаление колонок с утечкой данных, если они присутствуют в датасете
    cols_to_drop = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 3. Очистка и приведение числовых колонок (пример для bina.az)
    # Поля в датасете могут содержать текст, валюту (AZN) или пробелы.
    # Преобразуем ключевые колонки price, Sahə (площадь), Otaq sayı (комнаты) к numeric:
    numeric_cols = ["price", "Sahə", "Otaq sayı", "Mərtəbə"]
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                # Очищаем строки от лишних символов (букв, валют, пробелов)
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[^\d.]", "", regex=True)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Удаление строк с пропущенными критически важными значениями (например, цена и площадь)
    subset_dropna = [c for c in ["price", "Sahə"] if c in df.columns]
    if subset_dropna:
        df = df.dropna(subset=subset_dropna)

    # 5. Обработка выбросов (Outliers rule)
    # Например: отсекаем недвижимость с нереалистично маленькой/большой площадью или ценой
    if "price" in df.columns:
        df = df[(df["price"] > 1000) & (df["price"] < 10_000_000)]
    if "Sahə" in df.columns:
        df = df[(df["Sahə"] > 10) & (df["Sahə"] < 2000)]

    # 6. Обработка пропусков в категориальных колонках
    cat_cols = ["Kateqoriya", "Təmir", "Binanın növü", "location", "city"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot encoding),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    df = df.copy()

    if "price" not in df.columns:
        raise ValueError("Target column 'price' not found in dataframe.")

    y = df["price"].values
    
    # Исключаем целевую колонку из признаков
    X_df = df.drop(columns=["price"])

    # Кодирование категориальных признаков (One-Hot Encoding)
    X_encoded = pd.get_dummies(X_df, drop_first=True, dtype=float)
    
    feature_names = list(X_encoded.columns)
    X = X_encoded.values

    return X, y, feature_names


def make_tier_label(y_price: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, float]:
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).
    """
    y_price = np.asarray(y_price)
    if threshold is None:
        threshold = float(np.median(y_price))
    
    # premium = 1, standard = 0
    y_tier = (y_price > threshold).astype(int)
    return y_tier, threshold


def train_val_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    val_size: float = 0.15, 
    test_size: float = 0.15, 
    seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    np.random.seed(seed)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    test_count = int(n * test_size)
    val_count = int(n * val_size)

    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + val_count]
    train_idx = indices[test_count + val_count :]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    return X_tr, y_tr, X_val, y_val, X_te, y_te


def standardize(X_tr: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    """
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    # Предотвращение деления на ноль для константных признаков
    std[std == 0.0] = 1.0

    scaled_tr = (X_tr - mean) / std
    scaled_others = tuple((arr - mean) / std for arr in others)

    return (scaled_tr, *scaled_others)
