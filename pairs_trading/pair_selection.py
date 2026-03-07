from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .config import StrategyConfig

PairScore = Tuple[str, str, float, float, float]
# (asset_a, asset_b, corr, eg_pvalue, johansen_trace_stat)


def _safe_log(prices: pd.DataFrame) -> pd.DataFrame:
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive to take log.")
    return np.log(prices)


def find_candidate_pairs(prices: pd.DataFrame, cfg: StrategyConfig) -> List[PairScore]:
    if len(prices) < cfg.min_history:
        raise ValueError(f"Need at least {cfg.min_history} rows of data.")

    log_prices = _safe_log(prices)
    symbols = list(log_prices.columns)
    out: List[PairScore] = []

    for a, b in combinations(symbols, 2):
        pair = log_prices[[a, b]].dropna()
        if len(pair) < cfg.min_history:
            continue

        corr = pair[a].pct_change().corr(pair[b].pct_change())
        if pd.isna(corr) or corr < cfg.min_correlation:
            continue

        # Engle-Granger cointegration p-value
        try:
            _, pval, _ = coint(pair[a], pair[b])
        except Exception:
            continue

        # Johansen trace statistic (rank 0 test)
        # det_order=0 => constant term, k_ar_diff=1 => one lag difference
        try:
            joh = coint_johansen(pair[[a, b]], det_order=0, k_ar_diff=1)
            trace_stat = float(joh.lr1[0])
        except Exception:
            trace_stat = float("nan")

        out.append((a, b, float(corr), float(pval), trace_stat))

    # Sort by strongest EG p-value, then highest correlation
    out.sort(key=lambda x: (x[3], -x[2]))
    return out[: cfg.max_pairs_to_check]
