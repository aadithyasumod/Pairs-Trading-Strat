from __future__ import annotations

import argparse

from .data import load_intraday_pair_csv
from .intraday_config import IntradayConfig
from .intraday_engine import backtest_intraday_pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday pairs backtest (equities, Zerodha-style workflow)")
    p.add_argument("--csv", required=True, help="Intraday CSV with timestamp + two symbol columns")
    p.add_argument("--asset-a", required=True, help="First symbol, e.g. HDFCBANK")
    p.add_argument("--asset-b", required=True, help="Second symbol, e.g. ICICIBANK")
    p.add_argument("--plot", action="store_true", help="Plot equity, z-score, health")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = IntradayConfig()

    px = load_intraday_pair_csv(args.csv, args.asset_a, args.asset_b)
    res = backtest_intraday_pair(px[args.asset_a], px[args.asset_b], cfg)
    m = res.metrics

    print(f"Pair: {args.asset_a}/{args.asset_b}")
    print(f"Bars: {len(px)}")
    print(f"Trades: {m['n_trades']}")
    print(f"Win rate: {m['win_rate']:.1%}" if m["n_trades"] else "Win rate: N/A")
    print(f"Final PnL: {res.equity_curve.iloc[-1]:,.2f}")
    print(f"Total costs: {m['total_costs']:,.2f}")
    print(f"Approx intraday Sharpe (annualized): {m['sharpe']:.3f}")
    print(f"Max drawdown: {m['max_drawdown']:,.2f}")
    print(f"Average health score: {res.model['health'].mean():.3f}")
    if res.n_trades:
        print("Exits by reason: " + ", ".join(f"{k}={v}" for k, v in res.trades["exit_reason"].value_counts().items()))

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

        ax[0].plot(res.equity_curve.index, res.equity_curve.values, label="Equity")
        ax[0].set_title("Intraday Equity Curve")
        ax[0].grid(True, alpha=0.3)

        z = res.model["zscore"]
        ax[1].plot(z.index, z.values, label="Z-score")
        ax[1].axhline(cfg.z_entry, linestyle="--")
        ax[1].axhline(-cfg.z_entry, linestyle="--")
        ax[1].axhline(cfg.z_exit, linestyle=":")
        ax[1].axhline(-cfg.z_exit, linestyle=":")
        ax[1].set_title("Spread Z-score")
        ax[1].grid(True, alpha=0.3)

        health = res.model["health"]
        ax[2].plot(health.index, health.values, label="Health", color="tab:green")
        ax[2].axhline(cfg.health_pause_threshold, color="tab:red", linestyle="--")
        ax[2].axhline(cfg.health_exit_threshold, color="tab:red", linestyle=":")
        ax[2].set_title("Regime Health Score")
        ax[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
