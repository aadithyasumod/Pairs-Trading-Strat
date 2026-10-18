"""Spread model and the single rule book shared by every backtest and the live trader.

A pairs trade bets that the spread log(A) - alpha - beta * log(B) between two cointegrated
prices reverts to its mean. The z-score measures how stretched the spread is:

- enter when it is stretched (|z| >= z_entry) but not broken (|z| < z_stop)
- take profit once it has reverted (z back inside -z_exit / +z_exit, or through the mean)
- stop out if it keeps widening (adverse |z| >= z_stop), the regime breaks, or it overstays
  its expected reversion time
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hedge_ratio import rolling_ols_hedge_ratio
from .ou import estimate_half_life
from .position_sizing import hedge_ratio_ok
from .risk import max_holding_days_from_half_life

FLAT = 0
LONG_SPREAD = 1  # long A, short beta x B: the spread is too low and should rise
SHORT_SPREAD = -1  # short A, long beta x B: the spread is too high and should fall

# Exits that mean the relationship misbehaved; the pair sits out a cooldown after them.
COOLDOWN_EXITS = frozenset({"exit_stop", "exit_regime", "exit_time"})


@dataclass(frozen=True)
class SignalRules:
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5
    cooldown_bars: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.z_exit < self.z_entry < self.z_stop):
            raise ValueError("Thresholds must satisfy 0 <= z_exit < z_entry < z_stop.")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be non-negative.")


def build_spread(log_a: pd.Series, log_b: pd.Series, alpha: pd.Series, beta: pd.Series) -> pd.Series:
    return log_a - alpha - beta * log_b


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.where(std > 0)


def spread_model(
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_window: int,
    zscore_window: int,
    ou_window: int,
) -> pd.DataFrame:
    """Causal spread model for one pair: alpha, beta, spread, zscore and half_life per bar.

    The hedge ratio applied at bar t is fitted on the bars ending at t-1, so the spread at t is
    an out-of-sample residual. Every column at t uses only prices known at the close of t.
    """
    if len(prices_a) != len(prices_b):
        raise ValueError("Price series lengths must match.")
    if (prices_a <= 0).any() or (prices_b <= 0).any():
        raise ValueError("Prices must be positive to take logs.")

    log_a = np.log(prices_a.astype(float))
    log_b = pd.Series(np.log(prices_b.astype(float)).to_numpy(), index=log_a.index)

    hedge = rolling_ols_hedge_ratio(log_a, log_b, window=hedge_window).shift(1)
    spread = build_spread(log_a, log_b, hedge["alpha"], hedge["beta"])
    return pd.DataFrame(
        {
            "alpha": hedge["alpha"],
            "beta": hedge["beta"],
            "spread": spread,
            "zscore": rolling_zscore(spread, zscore_window),
            "half_life": estimate_half_life(spread, window=ou_window),
        },
        index=log_a.index,
    )


def decide_position(
    position: int,
    z: float,
    bars_held: int,
    max_hold: int,
    rules: SignalRules,
    *,
    can_enter: bool = True,
    regime_ok: bool = True,
    force_reason: str | None = None,
) -> tuple[int, str]:
    """Target position for this bar and why. Pure function of the current state.

    Reasons: "entry", "hold", "flat", "exit_mean_reversion", "exit_stop", "exit_regime",
    "exit_time", or the caller's `force_reason` (for example a session square-off).
    """
    z_ok = z is not None and math.isfinite(z)

    if position != FLAT:
        if force_reason:
            return FLAT, force_reason
        if z_ok:
            # position * z is negative while the spread is still stretched against us.
            if position * z <= -rules.z_stop:
                return FLAT, "exit_stop"
            # Reverted to within z_exit of the mean, or overshot straight through it.
            if position * z >= -rules.z_exit:
                return FLAT, "exit_mean_reversion"
        if not regime_ok:
            return FLAT, "exit_regime"
        if bars_held >= max_hold:
            return FLAT, "exit_time"
        return position, "hold"

    if force_reason or not can_enter or not z_ok:
        return FLAT, "flat"
    if rules.z_entry <= abs(z) < rules.z_stop:
        return (SHORT_SPREAD if z > 0 else LONG_SPREAD), "entry"
    return FLAT, "flat"


@dataclass
class PairStateMachine:
    """Holds the per-pair trading state and applies `decide_position` one bar at a time.

    The backtests and the live trader both drive this class, so a replay of live bars makes
    exactly the same decisions as the backtest.
    """

    rules: SignalRules
    max_hold_cap: int
    default_hold: int = 10
    direction: int = FLAT
    bars_held: int = 0
    max_hold: int = 0
    cooldown_left: int = 0

    def on_bar(
        self,
        z: float,
        half_life: float,
        beta: float,
        *,
        can_enter: bool = True,
        regime_ok: bool = True,
        force_reason: str | None = None,
    ) -> tuple[int, str]:
        if self.direction != FLAT:
            self.bars_held += 1

        cooling_down = False
        if self.direction == FLAT and self.cooldown_left > 0:
            self.cooldown_left -= 1
            cooling_down = True

        allowed = can_enter and not cooling_down and hedge_ratio_ok(float(beta))
        target, reason = decide_position(
            self.direction,
            z,
            self.bars_held,
            self.max_hold,
            self.rules,
            can_enter=allowed,
            regime_ok=regime_ok,
            force_reason=force_reason,
        )

        if target != self.direction:
            self.bars_held = 0
            if target == FLAT:
                if reason in COOLDOWN_EXITS:
                    self.cooldown_left = self.rules.cooldown_bars
            else:
                self.max_hold = max_holding_days_from_half_life(half_life, self.max_hold_cap, self.default_hold)
            self.direction = target
        return target, reason
