from __future__ import annotations

import math

import numpy as np
import pandas as pd



def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.pct_change().rolling(window).corr(b.pct_change())


def volatility_spike_ratio(spread: pd.Series, window: int) -> pd.Series:
    short = spread.diff().rolling(max(10, window // 4)).std(ddof=0)
    long = spread.diff().rolling(window).std(ddof=0)
    return short / long


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
    if not math.isfinite(corr_value):
        corr_term = 0.0
    else:
        corr_term = float(np.clip((corr_value - min_corr) / max(1e-9, 1.0 - min_corr), 0.0, 1.0))

    if not math.isfinite(beta_jump):
        beta_term = 0.0
    else:
        beta_term = float(np.clip(1.0 - (abs(beta_jump) / max_beta_jump), 0.0, 1.0))

    if not math.isfinite(vol_spike):
        vol_term = 0.0
    else:
        vol_term = float(np.clip(1.0 - max(0.0, vol_spike - 1.0) / max(1e-9, max_vol_spike_ratio - 1.0), 0.0, 1.0))

    return 0.5 * corr_term + 0.3 * beta_term + 0.2 * vol_term
