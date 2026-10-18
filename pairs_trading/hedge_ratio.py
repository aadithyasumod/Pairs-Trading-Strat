from __future__ import annotations

import pandas as pd


def rolling_ols_hedge_ratio(log_a: pd.Series, log_b: pd.Series, window: int = 120) -> pd.DataFrame:
    """Rolling OLS: log_a = alpha + beta * log_b + error.

    The fit at bar t uses the `window` bars ending at t (inclusive). Closed-form OLS with an
    intercept is beta = cov(a, b) / var(b) and alpha = mean(a) - beta * mean(b), which lets
    pandas compute every window in one vectorised pass.
    """
    if len(log_a) != len(log_b):
        raise ValueError("Series lengths must match.")
    if window < 2:
        raise ValueError("window must be at least 2 bars.")

    log_b = pd.Series(log_b.to_numpy(), index=log_a.index)
    var_b = log_b.rolling(window).var()
    cov_ab = log_a.rolling(window).cov(log_b)

    beta = cov_ab / var_b.where(var_b > 1e-14)
    alpha = log_a.rolling(window).mean() - beta * log_b.rolling(window).mean()
    return pd.DataFrame({"alpha": alpha, "beta": beta}, index=log_a.index)
