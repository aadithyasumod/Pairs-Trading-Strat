from __future__ import annotations

import numpy as np
import pandas as pd



def rolling_ols_hedge_ratio(log_a: pd.Series, log_b: pd.Series, window: int = 120) -> pd.DataFrame:
    """Rolling OLS: log_a = alpha + beta * log_b + error."""
    if len(log_a) != len(log_b):
        raise ValueError("Series lengths must match.")

    idx = log_a.index
    betas = np.full(len(idx), np.nan)
    alphas = np.full(len(idx), np.nan)

    for i in range(window - 1, len(idx)):
        ya = log_a.iloc[i - window + 1 : i + 1].values
        xb = log_b.iloc[i - window + 1 : i + 1].values

        X = np.column_stack([np.ones(window), xb])
        # Least squares solution
        coeff, *_ = np.linalg.lstsq(X, ya, rcond=None)
        alphas[i], betas[i] = coeff[0], coeff[1]

    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=idx)
