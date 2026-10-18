from __future__ import annotations

import math

import numpy as np
import pandas as pd

LN2 = math.log(2.0)


def _half_life_from_slope(b: float) -> float:
    """Half-life (bars) of the AR(1) fit ds_t = a + b * s_{t-1} + e_t.

    phi = 1 + b is the bar-to-bar persistence, so a deviation halves after -ln2 / ln(phi) bars.
    b >= 0 means no mean reversion. phi <= 0 is an overshooting (faster-than-one-bar) process,
    where ln(phi) is undefined, so fall back to the continuous-time approximation ln2 / -b.
    """
    if not math.isfinite(b) or b >= 0:
        return float("nan")
    phi = 1.0 + b
    if phi > 0:
        return -LN2 / math.log(phi)
    return LN2 / -b


def half_life(spread: pd.Series) -> float:
    """Single-window half-life of a spread, used when screening a formation period."""
    s = spread.dropna()
    if len(s) < 3:
        return float("nan")
    ds = s.diff().iloc[1:].to_numpy()
    lag = s.shift(1).iloc[1:].to_numpy()
    var = lag.var()
    if var <= 0:
        return float("nan")
    b = float(np.cov(ds, lag, ddof=0)[0, 1] / var)
    return _half_life_from_slope(b)


def estimate_half_life(spread: pd.Series, window: int = 120) -> pd.Series:
    """Rolling half-life. The value at bar t uses only the `window` spread values before t.

    Each window of `window` spread values gives window - 1 pairs of (ds_t, s_{t-1}); the slope b
    is their covariance over the variance of s_{t-1}, then the whole series is shifted one bar so
    the estimate at t never includes s_t.
    """
    if window < 3:
        raise ValueError("window must be at least 3 bars.")

    n = window - 1
    ds = spread.diff()
    lag = spread.shift(1)
    cov = ds.rolling(n, min_periods=n).cov(lag)
    var = lag.rolling(n, min_periods=n).var()
    b = (cov / var.where(var > 0)).shift(1)

    phi = 1.0 + b
    hl = pd.Series(np.nan, index=spread.index, dtype=float)
    smooth = (b < 0) & (phi > 0)
    overshoot = (b < 0) & (phi <= 0)
    hl[smooth] = -LN2 / np.log(phi[smooth])
    hl[overshoot] = LN2 / -b[overshoot]
    return hl
