from __future__ import annotations

import argparse

import numpy as np

from .backtest import run_pair_backtest
from .config import StrategyConfig
from .data import generate_synthetic_prices, load_price_csv
from .hedge_ratio import rolling_ols_hedge_ratio
from .ou import estimate_half_life
from .pair_selection import find_candidate_pairs
from .signals import build_spread, raw_signal_from_z, rolling_zscore



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NIFTY pairs trading starter")
    p.add_argument("--csv", type=str, default=None, help="Path to price CSV with Date + symbols")
    p.add_argument("--asset-a", type=str, default=None, help="First asset symbol")
    p.add_argument("--asset-b", type=str, default=None, help="Second asset symbol")
    p.add_argument("--plot", action="store_true", help="Plot equity curve")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StrategyConfig()

    data = load_price_csv(args.csv) if args.csv else generate_synthetic_prices()
    prices = data.prices

    if args.asset_a and args.asset_b:
        a, b = args.asset_a, args.asset_b
        if a not in prices.columns or b not in prices.columns:
            raise ValueError(f"Assets must be in CSV columns. Available: {list(prices.columns)}")
        chosen = (a, b, np.nan, np.nan, np.nan)
    else:
        pairs = find_candidate_pairs(prices, cfg)
        chosen = pairs[0] if pairs else (prices.columns[0], prices.columns[1], np.nan, np.nan, np.nan)

    a, b, corr, eg_pval, joh_trace = chosen
    print(f"Selected pair: {a} / {b}")
    print(f"Correlation (prefilter): {corr:.4f}" if np.isfinite(corr) else "Correlation: N/A")
    print(f"Engle-Granger p-value: {eg_pval:.6f}" if np.isfinite(eg_pval) else "EG p-value: N/A")
    print(f"Johansen trace stat: {joh_trace:.4f}" if np.isfinite(joh_trace) else "Johansen trace: N/A")

    pair_px = prices[[a, b]].dropna().copy()
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

    print("\nBacktest summary")
    print(f"Trades: {res.trades}")
    print(f"Sharpe: {res.sharpe:.3f}")
    print(f"Max drawdown (currency units): {res.max_drawdown:,.2f}")
    print(f"Final PnL (currency units): {res.equity_curve.iloc[-1]:,.2f}")

    if args.plot:
        import matplotlib.pyplot as plt

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
