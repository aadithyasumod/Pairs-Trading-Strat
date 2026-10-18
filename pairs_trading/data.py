from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PriceData:
    prices: pd.DataFrame


@dataclass(frozen=True)
class PriceDataFreshness:
    """The recency of a price panel used for monitoring."""

    last_observation: pd.Timestamp
    age_days: int


def require_fresh_price_data(
    prices: pd.DataFrame,
    max_age_days: int | None,
    *,
    as_of: pd.Timestamp | str | None = None,
) -> PriceDataFreshness:
    """Reject stale data before it can be presented as a live trading signal.

    Set ``max_age_days`` to ``None`` only for historical research or demos. Live
    monitoring should use a small calendar-day allowance so weekends and market
    holidays do not create false failures.
    """
    if prices.empty:
        raise ValueError("Price data is empty; cannot determine freshness.")
    if max_age_days is not None and max_age_days < 0:
        raise ValueError("max_data_age_days must be non-negative or None.")

    last_observation = pd.Timestamp(prices.index.max())
    if pd.isna(last_observation):
        raise ValueError("Price data has no valid observation date.")

    reference = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if reference.tzinfo is not None:
        reference = reference.tz_convert("UTC").tz_localize(None)
    if last_observation.tzinfo is not None:
        last_observation = last_observation.tz_convert("UTC").tz_localize(None)

    age_days = max(0, int((reference.normalize() - last_observation.normalize()).days))
    freshness = PriceDataFreshness(last_observation=last_observation, age_days=age_days)

    if max_age_days is not None and age_days > max_age_days:
        raise ValueError(
            "Price data is stale for live monitoring: "
            f"last observation is {last_observation.date()} ({age_days} calendar days old; "
            f"limit is {max_age_days}). Refresh the configured CSV or use a symbols_file."
        )

    return freshness


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

    # Keep sparse columns/rows for large universes; pair-level logic handles missing data.
    prices = df[price_cols].dropna(how="all").ffill()
    return PriceData(prices=prices)


def ou_path(n: int, half_life: float, stationary_std: float, rng: np.random.Generator) -> np.ndarray:
    """Discrete Ornstein-Uhlenbeck (AR(1)) path with a known half-life in bars."""
    phi = 0.5 ** (1.0 / half_life)
    shock_std = stationary_std * np.sqrt(1.0 - phi**2)
    shocks = rng.normal(0.0, shock_std, n)
    s = np.empty(n)
    s[0] = rng.normal(0.0, stationary_std)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + shocks[t]
    return s


def generate_synthetic_prices(
    n_days: int = 800,
    seed: int = 42,
    half_life: float = 8.0,
    beta: float = 0.9,
    spread_std: float = 0.03,
    end: str | pd.Timestamp | None = None,
) -> PriceData:
    """One cointegrated pair plus two unrelated random walks, for learning and testing.

    log(HDFCBANK) = alpha + beta * log(ICICIBANK) + s_t, where s_t is an OU spread with the given
    half-life (days). AXISBANK and SBIN are independent random walks that should never be traded.
    """
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    idx = pd.bdate_range(end=end_ts, periods=n_days)
    n = len(idx)

    log_b = np.log(1300.0) + np.cumsum(rng.normal(0.0, 0.015, n))
    alpha = np.log(1700.0) - beta * np.log(1300.0)
    log_a = alpha + beta * log_b + ou_path(n, half_life, spread_std, rng)
    log_c = np.log(1100.0) + np.cumsum(rng.normal(0.0, 0.017, n))
    log_d = np.log(800.0) + np.cumsum(rng.normal(0.0, 0.018, n))

    prices = pd.DataFrame(
        {
            "HDFCBANK": np.exp(log_a),
            "ICICIBANK": np.exp(log_b),
            "AXISBANK": np.exp(log_c),
            "SBIN": np.exp(log_d),
        },
        index=idx,
    )
    prices.index.name = "Date"
    return PriceData(prices=prices)


def generate_synthetic_universe(
    n_days: int = 1500,
    half_lives: tuple[float, ...] = (5.0, 10.0, 20.0),
    n_decoy_pairs: int = 2,
    n_singles: int = 4,
    seed: int = 7,
    end: str | pd.Timestamp | None = None,
) -> PriceData:
    """A small universe with known answers.

    - Pk_A / Pk_B: cointegrated pairs with the given spread half-lives (should be selected)
    - Dk_A / Dk_B: highly correlated but NOT cointegrated (spread is a random walk)
    - Sk: stocks that only share the market factor
    All names load on a common market factor, so correlation alone cannot separate them.
    """
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    idx = pd.bdate_range(end=end_ts, periods=n_days)
    n = len(idx)
    market = np.cumsum(rng.normal(0.0, 0.01, n))

    def stock(level: float, idio_vol: float) -> np.ndarray:
        return np.log(level) + rng.uniform(0.8, 1.2) * market + np.cumsum(rng.normal(0.0, idio_vol, n))

    cols: dict[str, np.ndarray] = {}
    for k, hl in enumerate(half_lives, start=1):
        log_b = stock(rng.uniform(200, 2000), 0.012)
        beta = rng.uniform(0.7, 1.3)
        cols[f"P{k}_A"] = rng.normal(0.0, 0.5) + beta * log_b + ou_path(n, hl, 0.02, rng)
        cols[f"P{k}_B"] = log_b
    for k in range(1, n_decoy_pairs + 1):
        shared = stock(rng.uniform(200, 2000), 0.012)
        cols[f"D{k}_A"] = shared + np.cumsum(rng.normal(0.0, 0.006, n))
        cols[f"D{k}_B"] = shared + np.cumsum(rng.normal(0.0, 0.006, n)) + rng.normal(0.0, 0.3)
    for k in range(1, n_singles + 1):
        cols[f"S{k}"] = stock(rng.uniform(200, 2000), 0.015)

    prices = pd.DataFrame({c: np.exp(v) for c, v in cols.items()}, index=idx)
    prices.index.name = "Date"
    return PriceData(prices=prices)


def generate_synthetic_intraday(
    n_sessions: int = 20,
    bar_minutes: int = 5,
    half_life_bars: float = 20.0,
    beta: float = 0.9,
    spread_std: float = 0.006,
    seed: int = 11,
    start: str = "2026-01-05",
    symbols: tuple[str, str] = ("HDFCBANK", "ICICIBANK"),
) -> pd.DataFrame:
    """Intraday bars (09:15 to 15:25 NSE session) for one cointegrated pair, with overnight gaps."""
    rng = np.random.default_rng(seed)
    bars_per_session = 375 // bar_minutes
    sessions = pd.bdate_range(start=start, periods=n_sessions)
    stamps = [
        day + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(minutes=bar_minutes * k)
        for day in sessions
        for k in range(bars_per_session)
    ]
    n = len(stamps)

    steps = rng.normal(0.0, 0.002, n)
    steps[::bars_per_session] += rng.normal(0.0, 0.008, n_sessions)  # overnight gaps
    log_b = np.log(1300.0) + np.cumsum(steps)
    alpha = np.log(1700.0) - beta * np.log(1300.0)
    log_a = alpha + beta * log_b + ou_path(n, half_life_bars, spread_std, rng)

    out = pd.DataFrame({symbols[0]: np.exp(log_a), symbols[1]: np.exp(log_b)}, index=pd.DatetimeIndex(stamps))
    out.index.name = "timestamp"
    return out


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
