from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from .data import load_intraday_wide_csv
from .intraday_config import IntradayConfig
from .intraday_engine import backtest_intraday_pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluate intraday pairs from wide CSV")
    p.add_argument("--csv", required=True, help="Intraday CSV with timestamp + many symbols")
    p.add_argument("--top", type=int, default=20, help="Show top N pairs")
    p.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    p.add_argument("--min-bars", type=int, default=500, help="Minimum bars required for a pair")
    return p.parse_args()


def _evaluate_pair(px: pd.DataFrame, a: str, b: str, cfg: IntradayConfig, min_bars: int) -> dict:
    pair_px = px[[a, b]].dropna()
    if len(pair_px) < min_bars:
        return {}

    res = backtest_intraday_pair(pair_px[a], pair_px[b], cfg)
    m = res.metrics
    return {
        "asset_a": a,
        "asset_b": b,
        "bars": len(pair_px),
        "trades": m["n_trades"],
        "win_rate": m["win_rate"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "total_costs": m["total_costs"],
        "final_pnl": m["total_pnl"],
        "avg_health": float(res.model["health"].mean()),
    }


def main() -> None:
    args = parse_args()
    cfg = IntradayConfig()
    px = load_intraday_wide_csv(args.csv)

    pairs = list(combinations(px.columns, 2))
    if not pairs:
        raise RuntimeError("Need at least 2 symbols in CSV.")

    rows = []
    for a, b in pairs:
        row = _evaluate_pair(px, a, b, cfg, args.min_bars)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid intraday pairs found after filters.")

    df = pd.DataFrame(rows).sort_values(["sharpe", "final_pnl"], ascending=[False, False]).reset_index(drop=True)

    print(df.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved full ranking to: {args.out}")


if __name__ == "__main__":
    main()
