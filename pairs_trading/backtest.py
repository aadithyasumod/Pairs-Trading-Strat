from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .risk import max_holding_days_from_half_life
from .signals import stop_trigger


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    pnl: pd.Series
    trades: int
    sharpe: float
    max_drawdown: float


def run_pair_backtest(
    returns_a: pd.Series,
    returns_b: pd.Series,
    zscore: pd.Series,
    raw_signal: pd.Series,
    beta: pd.Series,
    half_life: pd.Series,
    cfg: StrategyConfig,
) -> BacktestResult:
    idx = returns_a.index
    pos = pd.Series(0, index=idx, dtype=int)
    pnl = pd.Series(0.0, index=idx, dtype=float)

    current_pos = 0
    days_in_trade = 0
    trades = 0

    for t in range(1, len(idx)):
        i = idx[t]

        desired = int(raw_signal.loc[i]) if np.isfinite(raw_signal.loc[i]) else 0
        z = float(zscore.loc[i]) if np.isfinite(zscore.loc[i]) else np.nan
        b = float(beta.loc[i]) if np.isfinite(beta.loc[i]) else 1.0
        hl = float(half_life.loc[i]) if np.isfinite(half_life.loc[i]) else np.nan
        max_hold = max_holding_days_from_half_life(hl, cfg.max_holding_days_cap)

        # Stop logic for open positions
        if current_pos != 0:
            days_in_trade += 1
            if stop_trigger(z, cfg) or days_in_trade > max_hold:
                current_pos = 0
                days_in_trade = 0

        # If flat, allow new entry
        if current_pos == 0 and desired != 0:
            current_pos = desired
            days_in_trade = 1
            trades += 1

        pos.iloc[t] = current_pos

        # Pair return approximation:
        # long spread => +retA - beta*retB
        gross_ret = current_pos * (returns_a.iloc[t] - b * returns_b.iloc[t])

        # Cost when position changes
        changed = int(pos.iloc[t] != pos.iloc[t - 1])
        cost = changed * (cfg.fee_bps + cfg.slippage_bps) / 10_000

        pnl.iloc[t] = cfg.notional_per_trade * (gross_ret - cost)

    equity = pnl.cumsum()

    daily = pnl / max(cfg.notional_per_trade, 1.0)
    vol = daily.std(ddof=0)
    sharpe = float((daily.mean() / vol) * np.sqrt(252)) if vol > 1e-12 else 0.0

    running_max = equity.cummax()
    dd = equity - running_max
    max_dd = float(dd.min())

    return BacktestResult(
        equity_curve=equity,
        positions=pos,
        pnl=pnl,
        trades=trades,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )
