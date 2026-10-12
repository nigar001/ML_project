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

FIGURES_DIR = "report/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_eda_figures() -> None:
    print("Loading raw data for EDA...")
    df_raw = data_prep.load_raw()
    
    print("Cleaning data for EDA...")
    df_clean = data_prep.clean(df_raw)
    
    # 1. Price and Log-Price Distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    prices = df_clean["price"] / 1000
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
    plt.savefig(os.path.join(FIGURES_DIR, "price_distribution.png"), dpi=300)
    plt.close()

    # 2. Area vs Price
    if "Sahə" in df_clean.columns and "price" in df_clean.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df_clean["Sahə"], df_clean["price"] / 1000, alpha=0.3, color="teal", s=10)
        plt.title("Area vs Price")
        plt.xlabel("Area (m^2)")
        plt.ylabel("Price (k AZN)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "area_vs_price.png"), dpi=300)
        plt.close()

    # 3. Rooms Distribution
    if "Otaq sayı" in df_clean.columns:
        plt.figure(figsize=(6, 4))
        rooms = df_clean["Otaq sayı"][df_clean["Otaq sayı"] <= 7]
        plt.hist(rooms, bins=np.arange(1, 9) - 0.5, color="purple", edgecolor="black", rwidth=0.8)
        plt.title("Rooms Distribution")
        plt.xlabel("Number of Rooms")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "rooms_distribution.png"), dpi=300)
        plt.close()

    # 4. Location Coordinates (lat/lng scatter)
    if "lat" in df_clean.columns and "lng" in df_clean.columns:
        plt.figure(figsize=(6, 5))
        valid_coords = df_clean[(df_clean["lat"] > 38) & (df_clean["lat"] < 42) & (df_clean["lng"] > 44) & (df_clean["lng"] > 44)] # approximate Azerbaijan bounds if needed, or just plot all non-zero
        plt.scatter(df_clean["lng"], df_clean["lat"], alpha=0.2, color="orange", s=5)
        plt.title("Geographic Distribution (lat vs lng)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "geospatial_lat_lng.png"), dpi=300)
        plt.close()

    # 5. Price-Tier Class Balance
    y_tier, threshold = data_prep.make_tier_label(df_clean["price"].values)
    unique, counts = np.unique(y_tier, return_counts=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["Standard (0)", "Premium (1)"], counts, color=["gray", "green"], width=0.6)
    plt.title(f"Price-Tier Balance (Threshold: {threshold:,.0f} AZN)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "tier_class_balance.png"), dpi=300)
    plt.close()

    print("All EDA figures generated successfully!")


if __name__ == "__main__":
    generate_eda_figures()
