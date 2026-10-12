"""
synthetic.py — seeded stand-in listings for developing and testing the benchmark.

This is NOT project data. It exists so the benchmark code can be run and tested
before the real data stage is ready. Nothing produced from it may be reported
as a project result.

The columns imitate the kind of numeric features the real bina.az table will
have, and the price is built from them plus noise, so models have something
real to learn: bigger and more central listings cost more.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "area_m2", "rooms", "floor", "total_floors", "lat", "lng", "repaired", "mortgage",
]

# Rough centre of Baku, used only to make "distance from centre" matter.
CENTRE_LAT, CENTRE_LNG = 40.377, 49.850


def make_listings(n: int = 5000, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y_price, feature_names) for n fake listings. Same seed, same data."""
    rng = np.random.default_rng(seed)

    area = np.clip(rng.lognormal(mean=4.5, sigma=0.45, size=n), 20, 600)   # median ~90 m2
    rooms = np.clip(np.round(area / 35 + rng.normal(0, 0.6, size=n)), 1, 8)
    total_floors = rng.integers(5, 26, size=n).astype(float)
    floor = np.ceil(rng.random(n) * total_floors)                          # 1 .. total_floors
    lat = CENTRE_LAT + rng.normal(0, 0.05, size=n)
    lng = CENTRE_LNG + rng.normal(0, 0.08, size=n)
    repaired = (rng.random(n) < 0.8).astype(float)
    mortgage = (rng.random(n) < 0.33).astype(float)

    distance = np.sqrt((lat - CENTRE_LAT) ** 2 + (lng - CENTRE_LNG) ** 2)
    log_price = (
        8.2
        + 0.95 * np.log(area)          # price grows almost in proportion to area
        - 4.0 * distance               # cheaper away from the centre (non-linear in lat/lng)
        + 0.15 * repaired
        + 0.05 * mortgage
        - 0.08 * (floor == 1)          # ground floor is worth a little less
        + rng.normal(0, 0.25, size=n)  # everything the features do not explain
    )
    price = np.round(np.exp(log_price), -3)                                # asking prices are round

    X = np.column_stack([area, rooms, floor, total_floors, lat, lng, repaired, mortgage])
    return X, price, list(FEATURE_NAMES)
