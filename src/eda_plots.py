"""
eda_plots.py — Exploratory Data Analysis & figure generation for the report.
Role: Data Engineer & EDA Specialist
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data_prep

# Создаем папку для графиков, если её нет
FIGURES_DIR = "report/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_eda_figures() -> None:
    print("Loading raw data...")
    df_raw = data_prep.load_raw()
    
    print("Cleaning data...")
    df_clean = data_prep.clean(df_raw)
    
    # 1. График распределения цен (Price Distribution & Log-Price)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    prices = df_clean["price"] / 1000  # в тысячах AZN
    plt.hist(prices, bins=50, color="skyblue", edgecolor="black")
    plt.title("Price Distribution (k AZN)")
    plt.xlabel("Price")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    log_prices = np.log10(df_clean["price"])
    plt.hist(log_prices, bins=50, color="salmon", edgecolor="black")
    plt.title("Log10(Price) Distribution")
    plt.xlabel("Log10(Price)")
    plt.ylabel("Count")

    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "price_distribution.png")
    plt.savefig(path1, dpi=300)
    plt.close()
    print(f"Saved: {path1}")

    # 2. Зависимость цены от площади (Area vs Price)
    if "Sahə" in df_clean.columns and "price" in df_clean.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df_clean["Sahə"], df_clean["price"] / 1000, alpha=0.3, color="teal", s=10)
        plt.title("Area vs Price")
        plt.xlabel("Area (m^2)")
        plt.ylabel("Price (k AZN)")
        plt.tight_layout()
        path2 = os.path.join(FIGURES_DIR, "area_vs_price.png")
        plt.savefig(path2, dpi=300)
        plt.close()
        print(f"Saved: {path2}")

    print("EDA figures generated successfully!")


if __name__ == "__main__":
    generate_eda_figures()
