from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Tuple

import pandas as pd

from .batch_daily import _evaluate_pair
from .config import StrategyConfig
from .data import load_price_csv
from .pair_selection import correlation_candidates, filter_symbols, screen_pair

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


def _cointegration_screen(
    prices: pd.DataFrame,
    candidates: Iterable[Pair],
    p_threshold: float,
    min_history: int = 252,
) -> pd.DataFrame:
    rows = []
    start = time.time()

    for i, (a, b) in enumerate(candidates, 1):
        row = screen_pair(prices[a], prices[b])
        if row is not None and row["bars"] >= min_history and row["eg_pvalue"] <= p_threshold:
            rows.append(row)

        if i % 2000 == 0:
            elapsed = time.time() - start
            print(f"Cointegration checked: {i} pairs in {elapsed:.1f}s, passed={len(rows)}")

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["eg_pvalue", "corr"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig(min_history=args.min_history, min_correlation=args.min_corr)

    prices = load_price_csv(args.csv).prices
    prices = filter_symbols(prices, min_history=args.min_history)

    print(f"Input symbols after quality filter: {prices.shape[1]}")

    candidates: List[Pair] = correlation_candidates(
        prices,
        min_corr=args.min_corr,
        top_neighbors=args.top_neighbors,
        max_candidates=args.max_candidates,
    )
    print(f"Candidate pairs after correlation-neighbor prefilter: {len(candidates)}")

    if not candidates:
        raise RuntimeError("No pairs passed correlation prefilter. Reduce --min-corr.")

    coint_df = _cointegration_screen(prices, candidates, p_threshold=args.coint_p_threshold, min_history=args.min_history)
    print(f"Pairs after cointegration screen: {len(coint_df)}")

    if coint_df.empty:
        raise RuntimeError("No pairs passed cointegration filter. Relax --coint-p-threshold.")

    eval_df = coint_df.head(args.max_backtests)
    print(f"Running full backtests for top {len(eval_df)} cointegrated pairs...")

    rows = []
    for i, row in enumerate(eval_df.itertuples(index=False), 1):
        out = _evaluate_pair(prices, row.asset_a, row.asset_b, cfg)
        if out:
            out["corr"] = float(row.corr)
            out["eg_pvalue"] = float(row.eg_pvalue)
            out["half_life"] = float(row.half_life)
            rows.append(out)

        if i % 200 == 0:
            print(f"Backtests completed: {i}/{len(eval_df)}")

    if not rows:
        raise RuntimeError("No pairs produced valid backtest output.")

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["sharpe", "final_pnl"], ascending=[False, False]).reset_index(drop=True)

    print("\nTop ranked pairs (in-sample; validate with walk_forward before trusting):")
    print(out_df.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"\nSaved full ranking: {args.out}")


if __name__ == "__main__":
    main()
