from __future__ import annotations

import argparse
import time
from itertools import combinations
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .backtest import run_pair_backtest
from .config import StrategyConfig
from .data import load_price_csv
from .hedge_ratio import rolling_ols_hedge_ratio
from .ou import estimate_half_life
from .signals import build_spread, raw_signal_from_z, rolling_zscore

Pair = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scalable NSE universe pair scanner")
    p.add_argument("--csv", required=True, help="Wide daily CSV: Date + many symbols")
    p.add_argument("--top", type=int, default=50, help="Top N rows to print")
    p.add_argument("--out", type=str, default=None, help="Output CSV for full ranking")
    p.add_argument("--min-history", type=int, default=252, help="Minimum daily bars per symbol")
    p.add_argument("--min-corr", type=float, default=0.65, help="Minimum correlation for prefilter")
    p.add_argument("--top-neighbors", type=int, default=20, help="Keep top K correlated neighbors per symbol")
    p.add_argument(
        "--max-candidates",
        type=int,
        default=40000,
        help="Hard cap on candidate pairs after prefilter",
    )
    p.add_argument(
        "--max-backtests",
        type=int,
        default=2000,
        help="Run full backtest only for top M cointegrated pairs",
    )
    p.add_argument(
        "--coint-p-threshold",
        type=float,
        default=0.05,
        help="Engle-Granger p-value cutoff before backtest",
    )
    return p.parse_args()


def _filter_symbols(prices: pd.DataFrame, min_history: int) -> pd.DataFrame:
    # Keep symbols with enough valid points and positive prices
    good = []
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) >= min_history and (s > 0).all():
            good.append(c)
    if len(good) < 2:
        raise RuntimeError("Not enough valid symbols after filtering.")
    return prices[good].copy()


def _build_corr_candidates(
    prices: pd.DataFrame,
    min_corr: float,
    top_neighbors: int,
    max_candidates: int,
) -> List[Pair]:
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    corr = rets.corr(min_periods=60)

    symbols = list(corr.columns)
    pairs = set()

    for s in symbols:
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


def _cointegration_screen(
    prices: pd.DataFrame,
    candidates: Iterable[Pair],
    p_threshold: float,
) -> pd.DataFrame:
    rows = []
    start = time.time()

    for i, (a, b) in enumerate(candidates, 1):
        pair = prices[[a, b]].dropna()
        if len(pair) < 252:
            continue

        r = pair[a].pct_change().corr(pair[b].pct_change())
        if not np.isfinite(r):
            continue

        try:
            _, pval, _ = coint(np.log(pair[a]), np.log(pair[b]))
        except Exception:
            continue

        if pval <= p_threshold:
            rows.append({"asset_a": a, "asset_b": b, "corr": float(r), "eg_pvalue": float(pval)})

        if i % 2000 == 0:
            elapsed = time.time() - start
            print(f"Cointegration checked: {i} pairs in {elapsed:.1f}s, passed={len(rows)}")

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["eg_pvalue", "corr"], ascending=[True, False]).reset_index(drop=True)


def _evaluate_pair(prices: pd.DataFrame, a: str, b: str, cfg: StrategyConfig) -> dict:
    pair_px = prices[[a, b]].dropna()
    if len(pair_px) < cfg.min_history:
        return {}

    log_a = np.log(pair_px[a])
    log_b = np.log(pair_px[b])

    hr = rolling_ols_hedge_ratio(log_a, log_b, window=cfg.hedge_window)
    spread = build_spread(log_a, log_b, hr["alpha"], hr["beta"])
    z = rolling_zscore(spread, window=cfg.zscore_window)
    hl = estimate_half_life(spread, window=cfg.ou_window)
    sig = raw_signal_from_z(z, cfg)

    ret_a = pair_px[a].pct_change().fillna(0.0)
    ret_b = pair_px[b].pct_change().fillna(0.0)

    res = run_pair_backtest(
        returns_a=ret_a,
        returns_b=ret_b,
        zscore=z,
        raw_signal=sig,
        beta=hr["beta"].fillna(1.0),
        half_life=hl,
        cfg=cfg,
    )

    return {
        "asset_a": a,
        "asset_b": b,
        "bars": len(pair_px),
        "trades": int(res.trades),
        "sharpe": float(res.sharpe),
        "max_drawdown": float(res.max_drawdown),
        "final_pnl": float(res.equity_curve.iloc[-1]),
    }


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig(min_history=args.min_history, min_correlation=args.min_corr)

    prices = load_price_csv(args.csv).prices
    prices = _filter_symbols(prices, min_history=args.min_history)

    print(f"Input symbols after quality filter: {prices.shape[1]}")

    candidates = _build_corr_candidates(
        prices,
        min_corr=args.min_corr,
        top_neighbors=args.top_neighbors,
        max_candidates=args.max_candidates,
    )
    print(f"Candidate pairs after correlation-neighbor prefilter: {len(candidates)}")

    if not candidates:
        raise RuntimeError("No pairs passed correlation prefilter. Reduce --min-corr.")

    coint_df = _cointegration_screen(prices, candidates, p_threshold=args.coint_p_threshold)
    print(f"Pairs after cointegration screen: {len(coint_df)}")

    if coint_df.empty:
        raise RuntimeError("No pairs passed cointegration filter. Relax --coint-p-threshold.")

    eval_df = coint_df.head(args.max_backtests)
    print(f"Running full backtests for top {len(eval_df)} cointegrated pairs...")

    rows = []
    for i, row in eval_df.iterrows():
        out = _evaluate_pair(prices, row["asset_a"], row["asset_b"], cfg)
        if out:
            out["corr"] = float(row["corr"])
            out["eg_pvalue"] = float(row["eg_pvalue"])
            rows.append(out)

        if (i + 1) % 200 == 0:
            print(f"Backtests completed: {i + 1}/{len(eval_df)}")

    if not rows:
        raise RuntimeError("No pairs produced valid backtest output.")

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["sharpe", "final_pnl"], ascending=[False, False]).reset_index(drop=True)

    print("\nTop ranked pairs:")
    print(out_df.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"\nSaved full ranking: {args.out}")


if __name__ == "__main__":
    main()
