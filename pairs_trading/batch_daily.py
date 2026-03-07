from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from .backtest import run_pair_backtest
from .config import StrategyConfig
from .data import load_price_csv
from .hedge_ratio import rolling_ols_hedge_ratio
from .ou import estimate_half_life
from .pair_selection import find_candidate_pairs
from .signals import build_spread, raw_signal_from_z, rolling_zscore



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluate daily pairs from wide CSV")
    p.add_argument("--csv", required=True, help="Daily CSV with Date + many symbols")
    p.add_argument("--top", type=int, default=20, help="Show top N pairs")
    p.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    p.add_argument("--all-combinations", action="store_true", help="Evaluate all combinations (slower)")
    return p.parse_args()


def _evaluate_pair(prices: pd.DataFrame, a: str, b: str, cfg: StrategyConfig) -> dict:
    pair_px = prices[[a, b]].dropna().copy()
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
    cfg = StrategyConfig()
    prices = load_price_csv(args.csv).prices

    rows = []

    if args.all_combinations:
        symbols = list(prices.columns)
        pairs = list(combinations(symbols, 2))
    else:
        candidates = find_candidate_pairs(prices, cfg)
        pairs = [(a, b) for a, b, _, _, _ in candidates]

    if not pairs:
        raise RuntimeError("No candidate pairs found.")

    for a, b in pairs:
        row = _evaluate_pair(prices, a, b, cfg)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid pairs could be evaluated.")

    df = pd.DataFrame(rows).sort_values(["sharpe", "final_pnl"], ascending=[False, False]).reset_index(drop=True)

    print(df.head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved full ranking to: {args.out}")


if __name__ == "__main__":
    main()
