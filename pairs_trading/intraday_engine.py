"""Intraday (MIS) pair backtest: same rule book as daily, plus session rules.

- no new entries outside [no_new_entries_before, no_new_entries_after)
- every position is squared off at square_off_time, or on the session's last bar if earlier
- health below health_pause_threshold blocks entries; below the lower health_exit_threshold it
  also exits open trades (two thresholds, so a score hovering near one level cannot whipsaw)
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .backtest import PairBacktestResult, simulate_pair
from .intraday_config import IntradayConfig
from .regime import pair_health
from .signals import SignalRules, spread_model

SQUARE_OFF_REASON = "exit_square_off"


def _clock(value: str) -> dt.time:
    return dt.datetime.strptime(value, "%H:%M").time()


def intraday_rules(cfg: IntradayConfig) -> SignalRules:
    return SignalRules(cfg.z_entry, cfg.z_exit, cfg.z_stop, cfg.cooldown_bars_after_stop)


def bars_per_session(cfg: IntradayConfig) -> int:
    return max(1, 375 // max(1, cfg.bar_minutes))


def intraday_periods_per_year(cfg: IntradayConfig) -> float:
    return 252.0 * bars_per_session(cfg)


def pair_capital(cfg: IntradayConfig) -> float:
    return min(cfg.max_capital_per_pair, cfg.capital * cfg.max_gross_leverage / 2.0)


def build_intraday_model(prices_a: pd.Series, prices_b: pd.Series, cfg: IntradayConfig) -> pd.DataFrame:
    model = spread_model(prices_a, prices_b, cfg.hedge_window_bars, cfg.zscore_window_bars, cfg.ou_window_bars)
    model["health"] = pair_health(
        prices_a,
        prices_b,
        model,
        window=cfg.hedge_window_bars,
        min_corr=cfg.min_rolling_corr,
        max_beta_jump=cfg.max_beta_jump,
        max_vol_spike_ratio=cfg.max_vol_spike_ratio,
    )
    return model


def intraday_gates(
    index: pd.DatetimeIndex,
    health: pd.Series,
    cfg: IntradayConfig,
    detect_session_end: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-bar (can_enter, regime_ok, force_exit) for the intraday session rules.

    `detect_session_end` treats the last bar of each date as a square-off bar, which a backtest can
    see but a live trader cannot (it relies on square_off_time instead).
    """
    times = pd.Series(index.time, index=index)
    square_off = times >= _clock(cfg.square_off_time)
    if detect_session_end:
        dates = pd.Series(index.normalize(), index=index)
        square_off |= dates != dates.shift(-1)

    regime_ok = health >= cfg.health_exit_threshold
    in_entry_window = (times >= _clock(cfg.no_new_entries_before)) & (times < _clock(cfg.no_new_entries_after))
    can_enter = in_entry_window & (health >= cfg.health_pause_threshold) & ~square_off
    force_exit = pd.Series(np.where(square_off, SQUARE_OFF_REASON, None), index=index, dtype=object)
    return can_enter, regime_ok, force_exit


def backtest_intraday_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    cfg: IntradayConfig,
) -> PairBacktestResult:
    px = pd.concat([prices_a.rename("a"), prices_b.rename("b")], axis=1).dropna()
    if len(px) < cfg.hedge_window_bars + cfg.zscore_window_bars + 2:
        raise ValueError("Not enough intraday bars to warm up the hedge and z-score windows.")

    model = build_intraday_model(px["a"], px["b"], cfg)
    can_enter, regime_ok, force_exit = intraday_gates(px.index, model["health"], cfg)

    return simulate_pair(
        px["a"],
        px["b"],
        model,
        intraday_rules(cfg),
        notional=pair_capital(cfg),
        cost_bps_per_leg=cfg.fee_bps_per_leg + cfg.slippage_bps_per_leg,
        max_hold_cap=cfg.max_holding_bars_cap,
        default_hold=30,
        execution_lag=cfg.execution_lag_bars,
        can_enter=can_enter,
        regime_ok=regime_ok,
        force_exit=force_exit,
        capital=cfg.capital,
        periods_per_year=intraday_periods_per_year(cfg),
    )
