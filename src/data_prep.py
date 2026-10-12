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

# Explicit whitelists for features to prevent sparse matrix explosion and data leakage
NUMERIC_FEATURES = [
    "Sahə",
    "Otaq sayı",
    "floor",
    "total_floors",
    "Torpaq sahəsi",
    "lat",
    "lng",
]

CATEGORICAL_FEATURES = [
    "Kateqoriya",
    "Təmir",
    "Çıxarış",
    "İpoteka",
    "location",
    "city",
]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except FileNotFoundError:
        try:
            return pd.read_csv(path, encoding="latin1", low_memory=False)
        except Exception as e:
            raise FileNotFoundError(f"Could not load dataset from {path}. Error: {e}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw frame:
      - Remove duplicates based on listing URL (keeping the latest scrape),
      - Parse complex strings (floors '7 / 9', area units like 'sot' for both Sahə and Torpaq sahəsi),
      - Handle missing values safely (constant imputation to avoid test leakage),
      - Handle outliers with documented rules,
      - Drop leakage columns.
    """
    df = df.copy()

    # 1. Deduplication: same listing scraped on different days share 'estate_rel_url_x'
    if "estate_rel_url_x" in df.columns:
        df = df.drop_duplicates(subset=["estate_rel_url_x"], keep="last")
    else:
        df = df.drop_duplicates()

    # 2. Drop leakage columns
    cols_to_drop = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Clean target 'price'
    if "price" in df.columns:
        if df["price"].dtype == object:
            df["price"] = df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 4. Parse Floor ('Mərtəbə'): convert '7 / 9' into two numeric columns
    if "Mərtəbə" in df.columns:
        floor_split = df["Mərtəbə"].astype(str).str.split("/", expand=True)
        df["floor"] = pd.to_numeric(floor_split[0].str.replace(r"[^\d.]", "", regex=True), errors="coerce")
        df["total_floors"] = pd.to_numeric(floor_split[1].str.replace(r"[^\d.]", "", regex=True), errors="coerce") if floor_split.shape[1] > 1 else np.nan

    # 5. Clean area columns (Sahə & Torpaq sahəsi) with 'sot' / 'сот' unit conversion (*100 m2)
    for col in ["Sahə", "Torpaq sahəsi"]:
        if col in df.columns:
            s_str = df[col].astype(str)
            is_sot = s_str.str.contains("sot|сот", case=False, na=False)
            extracted = pd.to_numeric(s_str.str.replace(r"[^\d.]", "", regex=True), errors="coerce")
            df[col] = np.where(is_sot, extracted * 100.0, extracted)

    # 6. Clean other numeric columns
    for col in ["Otaq sayı", "lat", "lng", "floor", "total_floors"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7. Drop rows missing critical target or area
    required_cols = [c for c in ["price", "Sahə"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)

    # 8. Outlier filtering (documented rule):
    # - Price between 1,000 AZN and 10,000,000 AZN (removes extreme unpriced listings or data entry errors).
    # - Area between 15 m² and 3,000 m² (removes micro-studios or erroneous massive entries).
    if "price" in df.columns:
        df = df[(df["price"] >= 1000) & (df["price"] <= 10_000_000)]
    if "Sahə" in df.columns:
        df = df[(df["Sahə"] >= 15) & (df["Sahə"] <= 3000)]

    # 9. Handle missing values safely using constants (avoids test-set statistics leakage)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Turn the cleaned frame into a compact numeric design matrix using an explicit whitelist.
    Returns (X, y, feature_names).
    """
    df = df.copy()

    if "price" not in df.columns:
        raise ValueError("Target column 'price' not found in dataframe.")

    y = df["price"].values

    # Select only whitelisted features that exist in the dataframe
    active_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    active_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X_num = df[active_num].astype(float)
    
    if active_cat:
        X_cat = pd.get_dummies(df[active_cat], drop_first=True, dtype=float)
        X_combined = pd.concat([X_num, X_cat], axis=1)
    else:
        X_combined = X_num

    feature_names = list(X_combined.columns)
    X = X_combined.values

    return X, y, feature_names


def make_tier_label(y_price: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, float]:
    """
    Derived classification target: price TIER (premium vs standard).
    Uses the MEDIAN of the TRAINING price as the split threshold.
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
    seed: int = SEED,
    stratify_labels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic stratified/standard split into train / validation / test.
    Returns (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    np.random.seed(seed)
    n = len(X)
    indices = np.arange(n)

    if stratify_labels is not None:
        labels = np.asarray(stratify_labels)
        train_idx_list, val_idx_list, test_idx_list = [], [], []
        
        for cls in np.unique(labels):
            cls_indices = indices[labels == cls]
            np.random.shuffle(cls_indices)
            
            n_cls = len(cls_indices)
            test_c = int(n_cls * test_size)
            val_c = int(n_cls * val_size)
            
            test_idx_list.extend(cls_indices[:test_c])
            val_idx_list.extend(cls_indices[test_c : test_c + val_c])
            train_idx_list.extend(cls_indices[test_c + val_c:])
            
        train_idx = np.array(train_idx_list)
        val_idx = np.array(val_idx_list)
        test_idx = np.array(test_idx_list)
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)
    else:
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
