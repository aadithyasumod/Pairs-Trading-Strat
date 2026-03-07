from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PriceData:
    prices: pd.DataFrame


def load_price_csv(path: str) -> PriceData:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Keep only numeric price columns
    price_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not price_cols:
        raise ValueError("No numeric price columns found in CSV.")

    prices = df[price_cols].dropna(how="all").ffill().dropna(how="any")
    return PriceData(prices=prices)


def generate_synthetic_prices(n_days: int = 800, seed: int = 42) -> PriceData:
    """Generate simple synthetic data with one cointegrated pair for learning/testing."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    n = len(idx)

    # Build one strongly related pair using shared random-walk innovations.
    common_shock = rng.normal(0, 0.9, n)
    base = 100 + np.cumsum(common_shock)
    a = base + rng.normal(0, 0.15, n)
    b = 0.95 * base + 8 + rng.normal(0, 0.15, n)

    # Two unrelated controls.
    c = 50 + np.cumsum(rng.normal(0, 1.4, n))
    d = 70 + np.cumsum(rng.normal(0, 1.6, n))

    prices = pd.DataFrame(
        {
            "HDFCBANK": np.exp(a / 35),
            "ICICIBANK": np.exp(b / 35),
            "AXISBANK": np.exp(c / 35),
            "SBIN": np.exp(d / 35),
        },
        index=idx,
    )
    return PriceData(prices=prices)


def load_intraday_pair_csv(path: str, asset_a: str, asset_b: str) -> pd.DataFrame:
    """
    Expected CSV format:
      timestamp,HDFCBANK,ICICIBANK
      2026-01-02 09:15:00,1721.5,1298.2
      2026-01-02 09:20:00,1722.1,1298.0
    """
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' column for intraday mode.")
    for sym in [asset_a, asset_b]:
        if sym not in df.columns:
            raise ValueError(f"CSV missing required symbol column: {sym}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = (
        df[["timestamp", asset_a, asset_b]]
        .set_index("timestamp")
        .sort_index()
        .dropna(how="any")
        .astype(float)
    )
    return out


def load_intraday_wide_csv(path: str) -> pd.DataFrame:
    """
    Expected CSV format:
      timestamp,SYM1,SYM2,SYM3,...
    """
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' column for intraday mode.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric symbol columns.")

    out = df[numeric_cols].dropna(how="all").ffill().dropna(how="any").astype(float)
    return out
