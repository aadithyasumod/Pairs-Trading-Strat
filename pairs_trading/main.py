from __future__ import annotations

import argparse

import numpy as np

from .backtest import backtest_pair
from .config import StrategyConfig
from .data import generate_synthetic_prices, load_price_csv
from .pair_selection import find_candidate_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NIFTY pairs trading starter")
    p.add_argument("--csv", type=str, default=None, help="Path to price CSV with Date + symbols")
    p.add_argument("--asset-a", type=str, default=None, help="First asset symbol")
    p.add_argument("--asset-b", type=str, default=None, help="Second asset symbol")
    p.add_argument("--plot", action="store_true", help="Plot equity curve")
    return p.parse_args()


def _fmt(value: float, spec: str) -> str:
    return format(value, spec) if np.isfinite(value) else "N/A"


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig()

    data = load_price_csv(args.csv) if args.csv else generate_synthetic_prices()
    prices = data.prices

    if args.asset_a and args.asset_b:
        a, b = args.asset_a, args.asset_b
        if a not in prices.columns or b not in prices.columns:
            raise ValueError(f"Assets must be in CSV columns. Available: {list(prices.columns)}")
        stats = {"corr": np.nan, "eg_pvalue": np.nan, "johansen_trace": np.nan, "half_life": np.nan}
    else:
        pairs = find_candidate_pairs(prices, cfg)
        if pairs.empty:
            raise RuntimeError(
                "No pair passed the cointegration screen (EG p-value, positive beta, half-life bounds). "
                "Pass --asset-a/--asset-b to backtest a specific pair anyway."
            )
        stats = pairs.iloc[0].to_dict()
        a, b = stats["asset_a"], stats["asset_b"]

    print(f"Selected pair: {a} / {b}")
    print(f"Correlation (log returns): {_fmt(stats['corr'], '.4f')}")
    print(f"Engle-Granger p-value: {_fmt(stats['eg_pvalue'], '.6f')}")
    print(f"Johansen trace stat: {_fmt(stats['johansen_trace'], '.4f')}")
    print(f"Half-life (bars): {_fmt(stats['half_life'], '.2f')}")
    if not args.asset_a:
        print("Note: pair chosen and backtested on the same data (in-sample). Use walk_forward for out-of-sample.")

    res = backtest_pair(prices[a], prices[b], cfg)
    m = res.metrics

    print("\nBacktest summary")
    print(f"Trades: {m['n_trades']}")
    print(f"Win rate: {_fmt(m['win_rate'], '.1%')}")
    print(f"Profit factor: {_fmt(m['profit_factor'], '.2f')}")
    print(f"Avg holding (bars): {_fmt(m['avg_bars_held'], '.1f')}")
    print(f"Sharpe: {m['sharpe']:.3f}")
    print(f"Max drawdown (currency units): {m['max_drawdown']:,.2f}")
    print(f"Total costs (currency units): {m['total_costs']:,.2f}")
    print(f"Final PnL (currency units): {res.equity_curve.iloc[-1]:,.2f}")
    if res.n_trades:
        print("Exits by reason: " + ", ".join(f"{k}={v}" for k, v in res.trades["exit_reason"].value_counts().items()))

    if args.plot:
        import matplotlib.pyplot as plt

        z = res.model["zscore"]
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(res.equity_curve.index, res.equity_curve.values)
        ax[0].set_title("Equity Curve")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(z.index, z.values, label="z-score")
        ax[1].axhline(cfg.z_entry, linestyle="--")
        ax[1].axhline(-cfg.z_entry, linestyle="--")
        ax[1].axhline(cfg.z_exit, linestyle=":")
        ax[1].axhline(-cfg.z_exit, linestyle=":")
        ax[1].set_title("Spread Z-Score")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
