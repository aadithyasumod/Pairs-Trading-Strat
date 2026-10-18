"""Walk-forward (out-of-sample) pairs backtest, in the Gatev-Goetzmann-Rouwenhorst style.

Choosing pairs by their cointegration over a period and then backtesting them over that same
period is look-ahead: the test already knows which pairs will revert. This module instead rolls:

    formation window  [s, s + F)      -> screen and rank pairs using only this data
    trading window    [s + F, s + F + T) -> trade the selected pairs, then step s forward by T

Only trading-window PnL counts. The formation data warms up the rolling models (which are
causal), so the first trading bar already has a hedge ratio and z-score.

Run:
    python -m pairs_trading.walk_forward                      # synthetic universe demo
    python -m pairs_trading.walk_forward --csv daily_prices.csv --top-pairs 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import pandas as pd

from .backtest import backtest_pair, performance_metrics
from .config import StrategyConfig
from .data import generate_synthetic_universe, load_price_csv
from .pair_selection import find_candidate_pairs


@dataclass
class WalkForwardResult:
    pnl: pd.Series  # portfolio net PnL per trading-window bar
    trades: pd.DataFrame  # every out-of-sample trade, tagged with its window and pair
    windows: pd.DataFrame  # one row per formation/trading window
    metrics: dict  # out-of-sample portfolio metrics
    in_sample_metrics: dict  # the same pairs traded over their own formation windows


def _trade_pairs(
    prices: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: StrategyConfig,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    window_id: int,
) -> tuple[list[pd.Series], list[pd.DataFrame]]:
    pnls, trades = [], []
    for row in pairs.itertuples(index=False):
        try:
            res = backtest_pair(
                prices[row.asset_a], prices[row.asset_b], cfg, trade_start=trade_start, trade_end=trade_end
            )
        except ValueError:
            continue  # not enough overlapping history for this pair
        pnls.append(res.pnl.rename(f"{row.asset_a}/{row.asset_b}"))
        if len(res.trades):
            t = res.trades.copy()
            t.insert(0, "asset_b", row.asset_b)
            t.insert(0, "asset_a", row.asset_a)
            t.insert(0, "window", window_id)
            trades.append(t)
    return pnls, trades


def walk_forward(
    prices: pd.DataFrame,
    cfg: StrategyConfig | None = None,
    top_pairs: int | None = None,
    verbose: bool = False,
    top_neighbors: int = 20,
) -> WalkForwardResult:
    cfg = cfg or StrategyConfig()
    top_pairs = top_pairs or cfg.walk_forward_top_pairs
    formation, trading = cfg.formation_days, cfg.trading_days
    if formation < cfg.hedge_window + cfg.zscore_window + 2:
        raise ValueError("formation_days must cover the hedge and z-score warm-up windows.")
    if len(prices) < formation + 2:
        raise ValueError(f"Need more than {formation + 1} rows for a walk-forward test, got {len(prices)}.")

    # Screen with the formation length as the history requirement and keep the top N.
    select_cfg = replace(cfg, min_history=formation, max_pairs_to_check=top_pairs)
    idx = prices.index
    all_pnl: list[pd.Series] = []
    all_trades: list[pd.DataFrame] = []
    in_sample_pnl: list[pd.Series] = []
    in_sample_trades: list[pd.DataFrame] = []
    windows = []

    start, window_id = 0, 1
    while start + formation < len(idx):
        form_end = start + formation
        trade_end = min(form_end + trading, len(idx))
        form_px = prices.iloc[start:form_end].dropna(axis=1)
        trade_px = prices.iloc[start:trade_end]

        try:
            selected = find_candidate_pairs(form_px, select_cfg, top_neighbors=top_neighbors)
        except (RuntimeError, ValueError):
            selected = pd.DataFrame(columns=["asset_a", "asset_b"])

        pnls, trades = _trade_pairs(trade_px, selected, cfg, idx[form_end], idx[trade_end - 1], window_id)
        window_pnl = pd.concat(pnls, axis=1).sum(axis=1) if pnls else pd.Series(0.0, index=idx[form_end:trade_end])
        window_pnl = window_pnl.reindex(idx[form_end:trade_end], fill_value=0.0)
        all_pnl.append(window_pnl)
        all_trades.extend(trades)

        # In-sample comparison: the same pairs traded over the formation window that chose them.
        is_pnls, is_trades = _trade_pairs(
            form_px, selected, cfg, idx[start + cfg.hedge_window + cfg.zscore_window], idx[form_end - 1], window_id
        )
        if is_pnls:
            in_sample_pnl.append(pd.concat(is_pnls, axis=1).sum(axis=1))
        in_sample_trades.extend(is_trades)

        n_trades = int(sum(len(t) for t in trades))
        windows.append(
            {
                "window": window_id,
                "formation_start": idx[start],
                "trading_start": idx[form_end],
                "trading_end": idx[trade_end - 1],
                "pairs": ", ".join(f"{a}/{b}" for a, b in zip(selected["asset_a"], selected["asset_b"])),
                "n_pairs": int(len(selected)),
                "trades": n_trades,
                "pnl": float(window_pnl.sum()),
            }
        )
        if verbose:
            print(
                f"Window {window_id}: trade {idx[form_end].date()} -> {idx[trade_end - 1].date()} | "
                f"{len(selected)} pairs | {n_trades} trades | PnL {window_pnl.sum():,.0f}"
            )
        start += trading
        window_id += 1

    pnl = pd.concat(all_pnl).rename("pnl")
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    capital = cfg.notional_per_trade * top_pairs
    positions = pd.Series(0, index=pnl.index)  # portfolio exposure is not tracked per bar
    metrics = performance_metrics(pnl, trades_df if len(trades_df) else _empty_trades(), positions, capital, 252.0)
    metrics.pop("exposure")

    is_pnl = pd.concat(in_sample_pnl) if in_sample_pnl else pd.Series(dtype=float)
    is_trades = pd.concat(in_sample_trades, ignore_index=True) if in_sample_trades else _empty_trades()
    is_metrics = performance_metrics(is_pnl, is_trades, pd.Series(0, index=is_pnl.index), capital, 252.0)
    is_metrics.pop("exposure")
    metrics["walk_forward_efficiency"] = (
        metrics["ann_return"] / is_metrics["ann_return"] if is_metrics["ann_return"] > 0 else float("nan")
    )

    return WalkForwardResult(
        pnl=pnl,
        trades=trades_df,
        windows=pd.DataFrame(windows),
        metrics=metrics,
        in_sample_metrics=is_metrics,
    )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=["net_pnl", "return_on_gross", "bars_held", "costs"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward (out-of-sample) pairs trading backtest")
    p.add_argument("--csv", default=None, help="Wide daily CSV with Date + symbols (default: synthetic universe)")
    p.add_argument("--top-pairs", type=int, default=None, help="Pairs traded per window")
    p.add_argument("--formation-days", type=int, default=None)
    p.add_argument("--trading-days", type=int, default=None)
    p.add_argument("--top-neighbors", type=int, default=20, help="Correlation neighbours per symbol (large universes)")
    p.add_argument("--out-trades", default=None, help="Optional CSV for all out-of-sample trades")
    return p.parse_args()


def _print_metrics(title: str, m: dict) -> None:
    print(f"\n{title}")
    keys = [
        "total_pnl",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown_pct",
        "n_trades",
        "win_rate",
        "profit_factor",
        "avg_bars_held",
        "trade_t_stat",
        "total_costs",
    ]
    for k in keys + [k for k in ("walk_forward_efficiency",) if k in m]:
        v = m[k]
        print(f"  {k:24s} {v:,.4f}" if isinstance(v, float) else f"  {k:24s} {v}")


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig()
    if args.formation_days:
        cfg.formation_days = args.formation_days
    if args.trading_days:
        cfg.trading_days = args.trading_days

    if args.csv:
        prices = load_price_csv(args.csv).prices
        source = args.csv
    else:
        prices = generate_synthetic_universe().prices
        source = "synthetic universe (P* pairs are cointegrated, D* are decoys, S* are singles)"
    print(f"Data: {source} | {prices.shape[1]} symbols | {prices.index.min().date()} -> {prices.index.max().date()}")

    res = walk_forward(prices, cfg, top_pairs=args.top_pairs, verbose=True, top_neighbors=args.top_neighbors)
    _print_metrics("Out-of-sample portfolio (what you would have earned)", res.metrics)
    _print_metrics("In-sample (same pairs, their own formation window)", res.in_sample_metrics)

    if len(res.trades):
        print("\nOut-of-sample exits by reason:")
        print(res.trades["exit_reason"].value_counts().to_string())
    if args.out_trades and len(res.trades):
        res.trades.to_csv(args.out_trades, index=False)
        print(f"\nSaved trades: {args.out_trades}")


if __name__ == "__main__":
    main()
