from __future__ import annotations

import math

import numpy as np
import pandas as pd



def estimate_half_life(spread: pd.Series, window: int = 120) -> pd.Series:
    """Estimate rolling half-life using Δs_t = a + b*s_{t-1} + e_t."""
    hl = pd.Series(np.nan, index=spread.index, dtype=float)

    for i in range(window, len(spread)):
        s = spread.iloc[i - window : i].dropna()
        if len(s) < window:
            continue

        ds = s.diff().dropna().values
        s_lag = s.shift(1).dropna().values

        X = np.column_stack([np.ones(len(s_lag)), s_lag])
        coeff, *_ = np.linalg.lstsq(X, ds, rcond=None)
        b = coeff[1]

        if b < 0:
            kappa = -b
            hl.iloc[i] = math.log(2) / kappa if kappa > 1e-12 else np.nan

    return hl
