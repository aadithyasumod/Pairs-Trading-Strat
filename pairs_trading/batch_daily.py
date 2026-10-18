from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from .backtest import backtest_pair
from .config import StrategyConfig
from .data import load_price_csv
from .pair_selection import find_candidate_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluate daily pairs from wide CSV (in-sample ranking)")
    p.add_argument("--csv", required=True, help="Daily CSV with Date + many symbols")
    p.add_argument("--top", type=int, default=20, help="Show top N pairs")
    p.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    p.add_argument("--all-combinations", action="store_true", help="Evaluate all combinations (slower)")
    return p.parse_args()


def _evaluate_pair(prices: pd.DataFrame, a: str, b: str, cfg: StrategyConfig) -> dict:
    pair_px = prices[[a, b]].dropna()
    if len(pair_px) < cfg.min_history:
        return {}
    res = backtest_pair(pair_px[a], pair_px[b], cfg)
    m = res.metrics
    return {
        "asset_a": a,
        "asset_b": b,
        "bars": len(pair_px),
        "trades": m["n_trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "total_costs": m["total_costs"],
        "final_pnl": m["total_pnl"],
    }


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig()
    prices = load_price_csv(args.csv).prices

    if args.all_combinations:
        pairs = list(combinations(prices.columns, 2))
    else:
        candidates = find_candidate_pairs(prices, cfg)
        pairs = list(zip(candidates["asset_a"], candidates["asset_b"]))

    if not pairs:
        raise RuntimeError("No candidate pairs passed the cointegration screen. Try --all-combinations.")

    rows = []
    for a, b in pairs:
        row = _evaluate_pair(prices, a, b, cfg)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid pairs could be evaluated.")

    df = pd.DataFrame(rows).sort_values(["sharpe", "final_pnl"], ascending=[False, False]).reset_index(drop=True)

    print("In-sample ranking: pairs are screened and backtested on the same data. Validate with walk_forward.")
    print(df.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved full ranking to: {args.out}")


if __name__ == "__main__":
    main()
