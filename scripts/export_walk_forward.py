"""Run a walk-forward backtest and export it as an Excel workbook for manual review.

    .venv/bin/python scripts/export_walk_forward.py --csv daily_prices_nse_universe.csv --top-pairs 10

Sheets: metrics (out-of-sample vs in-sample), windows, trades, by_pair, by_exit_reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pairs_trading.config import StrategyConfig  # noqa: E402
from pairs_trading.data import generate_synthetic_universe, load_price_csv  # noqa: E402
from pairs_trading.walk_forward import walk_forward  # noqa: E402

TRADE_VIEW = [
    "window",
    "asset_a",
    "asset_b",
    "side",
    "entry_time",
    "exit_time",
    "bars_held",
    "exit_reason",
    "entry_z",
    "exit_z",
    "hedge_beta",
    "qty_a",
    "qty_b",
    "entry_price_a",
    "exit_price_a",
    "entry_price_b",
    "exit_price_b",
    "gross_notional",
    "gross_pnl",
    "costs",
    "net_pnl",
    "return_on_gross",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a walk-forward backtest to Excel")
    p.add_argument("--csv", default=None, help="Wide daily CSV (default: synthetic universe)")
    p.add_argument("--top-pairs", type=int, default=10)
    p.add_argument("--top-neighbors", type=int, default=20)
    p.add_argument("--formation-days", type=int, default=None)
    p.add_argument("--trading-days", type=int, default=None)
    p.add_argument("--out", default=None, help="Output .xlsx path (default: results/walk_forward_<date>.xlsx)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig()
    if args.formation_days:
        cfg.formation_days = args.formation_days
    if args.trading_days:
        cfg.trading_days = args.trading_days

    if args.csv:
        prices = load_price_csv(args.csv).prices
        source = Path(args.csv).name
    else:
        prices = generate_synthetic_universe().prices
        source = "synthetic universe"
    print(f"Data: {source} | {prices.shape[1]} symbols | {prices.index.min().date()} -> {prices.index.max().date()}")

    res = walk_forward(prices, cfg, top_pairs=args.top_pairs, verbose=True, top_neighbors=args.top_neighbors)
    trades = res.trades.copy()
    if trades.empty:
        raise SystemExit("No out-of-sample trades were generated; nothing to export.")

    trades["side"] = trades["direction"].map({1: "LONG_SPREAD (long A, short B)", -1: "SHORT_SPREAD (short A, long B)"})
    trades["pair"] = trades["asset_a"] + "/" + trades["asset_b"]
    trades = trades.sort_values("entry_time").reset_index(drop=True)

    run_info = {
        "data_file": source,
        "symbols": int(prices.shape[1]),
        "history_start": str(prices.index.min().date()),
        "history_end": str(prices.index.max().date()),
        "formation_days": cfg.formation_days,
        "trading_days": cfg.trading_days,
        "pairs_per_window": args.top_pairs,
        "notional_per_pair": cfg.notional_per_trade,
        "capital": cfg.notional_per_trade * args.top_pairs,
        "cost_bps_per_leg_per_side": cfg.fee_bps + cfg.slippage_bps,
        "execution_lag_bars": cfg.execution_lag_bars,
        "z_entry/z_exit/z_stop": f"{cfg.z_entry}/{cfg.z_exit}/{cfg.z_stop}",
        "coint_p_threshold": cfg.coint_p_threshold,
        "half_life_bounds": f"{cfg.min_half_life}-{cfg.max_half_life}",
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    metrics = pd.DataFrame(
        {
            "metric": list(res.metrics.keys()),
            "out_of_sample": list(res.metrics.values()),
            "in_sample": [res.in_sample_metrics.get(k) for k in res.metrics],
        }
    )
    settings = pd.DataFrame({"setting": list(run_info), "value": [str(v) for v in run_info.values()]})

    by_pair = (
        trades.groupby("pair")
        .agg(trades=("net_pnl", "size"), wins=("net_pnl", lambda s: int((s > 0).sum())),
             gross_pnl=("gross_pnl", "sum"), costs=("costs", "sum"), net_pnl=("net_pnl", "sum"))
        .sort_values("net_pnl")
        .reset_index()
    )
    by_reason = (
        trades.groupby("exit_reason")
        .agg(trades=("net_pnl", "size"), avg_net_pnl=("net_pnl", "mean"), total_net_pnl=("net_pnl", "sum"),
             avg_bars_held=("bars_held", "mean"))
        .sort_values("total_net_pnl")
        .reset_index()
    )
    equity = pd.DataFrame({"date": res.pnl.index, "daily_pnl": res.pnl.to_numpy(), "cumulative_pnl": res.pnl.cumsum().to_numpy()})

    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"walk_forward_{pd.Timestamp.now():%Y%m%d}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        settings.to_excel(writer, sheet_name="run_settings", index=False)
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        res.windows.to_excel(writer, sheet_name="windows", index=False)
        trades[TRADE_VIEW].to_excel(writer, sheet_name="trades", index=False)
        by_pair.to_excel(writer, sheet_name="by_pair", index=False)
        by_reason.to_excel(writer, sheet_name="by_exit_reason", index=False)
        equity.to_excel(writer, sheet_name="equity_curve", index=False)

    print(f"\nSaved workbook: {out}")
    print(f"Sheets: run_settings, metrics, windows, trades ({len(trades)}), by_pair ({len(by_pair)}), by_exit_reason, equity_curve")


if __name__ == "__main__":
    main()
