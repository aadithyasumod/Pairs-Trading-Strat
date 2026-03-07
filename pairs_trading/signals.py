from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig



def build_spread(log_a: pd.Series, log_b: pd.Series, alpha: pd.Series, beta: pd.Series) -> pd.Series:
    return log_a - alpha - beta * log_b


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std


def raw_signal_from_z(z: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """
    +1 => long spread (long A, short B)
    -1 => short spread (short A, long B)
     0 => flat
    """
    sig = pd.Series(0, index=z.index, dtype=int)
    sig[z <= -cfg.z_entry] = 1
    sig[z >= cfg.z_entry] = -1

    # Exit condition overrides entry if close to mean
    sig[z.abs() <= cfg.z_exit] = 0
    return sig


def stop_trigger(z: float, cfg: StrategyConfig) -> bool:
    return bool(np.isfinite(z) and abs(z) >= cfg.z_stop)
