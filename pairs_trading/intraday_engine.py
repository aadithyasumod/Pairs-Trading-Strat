from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hedge_ratio import rolling_ols_hedge_ratio
from .intraday_config import IntradayConfig
from .ou import estimate_half_life
from .position_sizing import PairOrderSize, compute_pair_order_size
from .regime import health_score, rolling_corr, volatility_spike_ratio
from .signals import build_spread, raw_signal_from_z, rolling_zscore


@dataclass
class IntradayBacktestResult:
    equity_curve: pd.Series
    pnl: pd.Series
    position: pd.Series
    health: pd.Series
    zscore: pd.Series
    trades: int



def _max_holding_bars(hl_bars: float, cap: int) -> int:
    if not np.isfinite(hl_bars) or hl_bars <= 0:
        return min(30, cap)
    return min(int(np.ceil(2.0 * hl_bars)), cap)


def _trade_cost_fraction(cfg: IntradayConfig) -> float:
    # 2 legs in a pair
    return 2.0 * (cfg.fee_bps_per_leg + cfg.slippage_bps_per_leg) / 10_000.0


def backtest_intraday_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    cfg: IntradayConfig,
) -> IntradayBacktestResult:
    px = pd.concat([prices_a.rename("a"), prices_b.rename("b")], axis=1).dropna()
    log_a = np.log(px["a"])
    log_b = np.log(px["b"])

    hr = rolling_ols_hedge_ratio(log_a, log_b, window=cfg.hedge_window_bars)
    spread = build_spread(log_a, log_b, hr["alpha"], hr["beta"])
    z = rolling_zscore(spread, cfg.zscore_window_bars)
    raw_sig = raw_signal_from_z(
        z,
        type("TmpCfg", (), {"z_entry": cfg.z_entry, "z_exit": cfg.z_exit, "z_stop": cfg.z_stop}),
    )
    hl = estimate_half_life(spread, cfg.ou_window_bars)

    corr = rolling_corr(px["a"], px["b"], cfg.hedge_window_bars)
    beta_jump = hr["beta"].diff().abs()
    vol_spike = volatility_spike_ratio(spread, cfg.hedge_window_bars)

    health = pd.Series(index=px.index, dtype=float)
    for i in px.index:
        health.loc[i] = health_score(
            corr_value=float(corr.loc[i]) if np.isfinite(corr.loc[i]) else np.nan,
            beta_jump=float(beta_jump.loc[i]) if np.isfinite(beta_jump.loc[i]) else np.nan,
            vol_spike=float(vol_spike.loc[i]) if np.isfinite(vol_spike.loc[i]) else np.nan,
            min_corr=cfg.min_rolling_corr,
            max_beta_jump=cfg.max_beta_jump,
            max_vol_spike_ratio=cfg.max_vol_spike_ratio,
        )

    ret_a = px["a"].pct_change().fillna(0.0)
    ret_b = px["b"].pct_change().fillna(0.0)

    pos = pd.Series(0, index=px.index, dtype=int)
    pnl = pd.Series(0.0, index=px.index, dtype=float)

    current_pos = 0
    bars_in_trade = 0
    cooldown_left = 0
    trades = 0

    pair_capital = min(cfg.max_capital_per_pair, cfg.capital * cfg.max_gross_leverage / 2.0)
    cost_frac = _trade_cost_fraction(cfg)

    for t in range(1, len(px.index)):
        i = px.index[t]

        beta_now = float(hr["beta"].iloc[t]) if np.isfinite(hr["beta"].iloc[t]) else 1.0
        hl_now = float(hl.iloc[t]) if np.isfinite(hl.iloc[t]) else np.nan
        max_hold = _max_holding_bars(hl_now, cfg.max_holding_bars_cap)

        if current_pos != 0:
            bars_in_trade += 1

        z_now = float(z.iloc[t]) if np.isfinite(z.iloc[t]) else np.nan
        health_now = float(health.iloc[t]) if np.isfinite(health.iloc[t]) else 0.0

        forced_exit = False
        if current_pos != 0 and np.isfinite(z_now) and abs(z_now) >= cfg.z_stop:
            forced_exit = True
        if current_pos != 0 and bars_in_trade > max_hold:
            forced_exit = True
        if current_pos != 0 and health_now < cfg.health_pause_threshold:
            forced_exit = True

        if forced_exit:
            current_pos = 0
            bars_in_trade = 0
            cooldown_left = cfg.cooldown_bars_after_stop

        desired = int(raw_sig.iloc[t]) if np.isfinite(raw_sig.iloc[t]) else 0

        if cooldown_left > 0:
            desired = 0
            cooldown_left -= 1

        if health_now < cfg.health_pause_threshold:
            desired = 0

        if current_pos == 0 and desired != 0:
            current_pos = desired
            bars_in_trade = 1
            trades += 1

        pos.iloc[t] = current_pos

        gross_ret = current_pos * (ret_a.iloc[t] - beta_now * ret_b.iloc[t])
        changed = int(pos.iloc[t] != pos.iloc[t - 1])

        size: PairOrderSize = compute_pair_order_size(
            price_a=float(px["a"].iloc[t]),
            price_b=float(px["b"].iloc[t]),
            beta=beta_now,
            pair_capital=pair_capital,
        )

        pnl.iloc[t] = size.gross_notional * gross_ret - changed * size.gross_notional * cost_frac

    equity = pnl.cumsum()
    return IntradayBacktestResult(
        equity_curve=equity,
        pnl=pnl,
        position=pos,
        health=health,
        zscore=z,
        trades=trades,
    )
