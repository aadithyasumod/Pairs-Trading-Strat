"""Pair discovery: data-quality filter -> correlation prefilter -> cointegration screen.

Correlation is only a cheap prefilter. The property a pairs trade relies on is cointegration:
a linear combination of the two log prices is stationary, so the spread keeps returning to its
mean. The Engle-Granger test checks that the OLS residual of log A on log B is stationary, and
the half-life of that residual says whether it reverts fast enough to trade.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .config import StrategyConfig
from .ou import half_life

Pair = Tuple[str, str]

PAIR_COLUMNS = ["asset_a", "asset_b", "corr", "eg_pvalue", "johansen_trace", "beta", "half_life", "bars"]

# Above this many symbols, testing every combination is too slow; use nearest-neighbour candidates.
ALL_PAIRS_SYMBOL_LIMIT = 60


def filter_symbols(prices: pd.DataFrame, min_history: int) -> pd.DataFrame:
    """Keep symbols with at least `min_history` valid, strictly positive prices."""
    keep = []
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) >= min_history and (s > 0).all():
            keep.append(c)
    if len(keep) < 2:
        raise RuntimeError("Not enough symbols after data-quality filter.")
    return prices[keep].copy()


def correlation_candidates(
    prices: pd.DataFrame,
    min_corr: float,
    top_neighbors: int,
    max_candidates: int,
) -> List[Pair]:
    """Each symbol's `top_neighbors` most correlated partners (daily log returns >= min_corr)."""
    rets = np.log(prices).diff().replace([np.inf, -np.inf], np.nan)
    corr = rets.corr(min_periods=60)

    pairs = set()
    for s in corr.columns:
        row = corr.loc[s].drop(labels=[s]).dropna()
        row = row[row >= min_corr].sort_values(ascending=False).head(top_neighbors)
        for t in row.index:
            a, b = sorted((s, t))
            pairs.add((a, b))
            if len(pairs) >= max_candidates:
                break
        if len(pairs) >= max_candidates:
            break

    return sorted(pairs)


def screen_pair(price_a: pd.Series, price_b: pd.Series, min_corr: float | None = None) -> dict | None:
    """Cointegration statistics for one pair over the given window, or None if untestable.

    beta and half_life come from a static OLS of log A on log B over the whole window, which is
    the same regression the Engle-Granger test runs.
    """
    pair = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1).dropna()
    if len(pair) < 30 or (pair <= 0).any().any():
        return None
    log_pair = np.log(pair)

    corr = float(log_pair["a"].diff().corr(log_pair["b"].diff()))
    if not np.isfinite(corr):
        return None
    if min_corr is not None and corr < min_corr:
        return None

    try:
        _, pval, _ = coint(log_pair["a"], log_pair["b"])
    except Exception:
        return None

    try:
        joh = coint_johansen(log_pair[["a", "b"]], det_order=0, k_ar_diff=1)
        trace_stat = float(joh.lr1[0])
    except Exception:
        trace_stat = float("nan")

    beta, alpha = np.polyfit(log_pair["b"].to_numpy(), log_pair["a"].to_numpy(), 1)
    residual = log_pair["a"] - alpha - beta * log_pair["b"]

    return {
        "asset_a": price_a.name,
        "asset_b": price_b.name,
        "corr": corr,
        "eg_pvalue": float(pval),
        "johansen_trace": trace_stat,
        "beta": float(beta),
        "half_life": half_life(residual),
        "bars": int(len(pair)),
    }


def passes_screen(row: dict, cfg: StrategyConfig) -> bool:
    """Cointegrated, positively hedged, and reverting on a tradeable time scale."""
    hl = row["half_life"]
    return (
        row["eg_pvalue"] <= cfg.coint_p_threshold
        and row["beta"] > 0
        and np.isfinite(hl)
        and cfg.min_half_life <= hl <= cfg.max_half_life
    )


def find_candidate_pairs(
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    candidates: List[Pair] | None = None,
    top_neighbors: int = 20,
    max_candidates: int = 40_000,
) -> pd.DataFrame:
    """Screen a price panel and return passing pairs ranked by Engle-Granger p-value."""
    if len(prices) < cfg.min_history:
        raise ValueError(f"Need at least {cfg.min_history} rows of data.")

    usable = filter_symbols(prices, cfg.min_history)
    if candidates is None:
        if usable.shape[1] <= ALL_PAIRS_SYMBOL_LIMIT:
            candidates = list(combinations(usable.columns, 2))
        else:
            candidates = correlation_candidates(usable, cfg.min_correlation, top_neighbors, max_candidates)

    rows = []
    for a, b in candidates:
        if a not in usable.columns or b not in usable.columns:
            continue
        row = screen_pair(usable[a], usable[b], min_corr=cfg.min_correlation)
        if row is not None and row["bars"] >= cfg.min_history and passes_screen(row, cfg):
            rows.append(row)

    out = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    out = out.sort_values(["eg_pvalue", "corr"], ascending=[True, False]).reset_index(drop=True)
    return out.head(cfg.max_pairs_to_check)
