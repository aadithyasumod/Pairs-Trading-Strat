from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.pct_change().rolling(window).corr(b.pct_change())


def volatility_spike_ratio(spread: pd.Series, window: int) -> pd.Series:
    """Short-horizon spread volatility relative to its longer-run level (1.0 = normal)."""
    short = spread.diff().rolling(max(10, window // 4)).std(ddof=0)
    long = spread.diff().rolling(window).std(ddof=0)
    return short / long.where(long > 0)


def _health_terms(corr, beta_jump, vol_spike, min_corr, max_beta_jump, max_vol_spike_ratio) -> np.ndarray:
    """0..1 stability score; any missing input scores zero for its term."""
    corr = np.asarray(corr, dtype=float)
    beta_jump = np.asarray(beta_jump, dtype=float)
    vol_spike = np.asarray(vol_spike, dtype=float)

    corr_term = np.clip((corr - min_corr) / max(1e-9, 1.0 - min_corr), 0.0, 1.0)
    beta_term = np.clip(1.0 - np.abs(beta_jump) / max_beta_jump, 0.0, 1.0)
    vol_term = np.clip(1.0 - np.maximum(0.0, vol_spike - 1.0) / max(1e-9, max_vol_spike_ratio - 1.0), 0.0, 1.0)

    score = (
        0.5 * np.nan_to_num(corr_term, nan=0.0)
        + 0.3 * np.nan_to_num(beta_term, nan=0.0)
        + 0.2 * np.nan_to_num(vol_term, nan=0.0)
    )
    return score


def health_score(
    corr_value: float,
    beta_jump: float,
    vol_spike: float,
    min_corr: float,
    max_beta_jump: float,
    max_vol_spike_ratio: float,
) -> float:
    """
    Returns a 0..1 score. Higher means pair relationship is more stable.
    This is intentionally simple and beginner-friendly.
    """
    return float(_health_terms(corr_value, beta_jump, vol_spike, min_corr, max_beta_jump, max_vol_spike_ratio))


def pair_health(
    price_a: pd.Series,
    price_b: pd.Series,
    model: pd.DataFrame,
    window: int,
    min_corr: float,
    max_beta_jump: float,
    max_vol_spike_ratio: float,
) -> pd.Series:
    """Health score for every bar of a pair, from its prices and its spread model."""
    corr = rolling_corr(price_a, price_b, window)
    beta_jump = model["beta"].diff().abs()
    vol_spike = volatility_spike_ratio(model["spread"], window)
    score = _health_terms(corr, beta_jump, vol_spike, min_corr, max_beta_jump, max_vol_spike_ratio)
    return pd.Series(score, index=model.index, name="health")
