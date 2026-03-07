from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .hedge_ratio import rolling_ols_hedge_ratio
from .intraday_config import IntradayConfig
from .position_sizing import compute_pair_order_size
from .regime import health_score, rolling_corr, volatility_spike_ratio
from .signals import build_spread, raw_signal_from_z, rolling_zscore
from .zerodha_adapter import OrderIntent, ZerodhaAdapter


@dataclass
class TraderState:
    position: int = 0  # +1 long spread, -1 short spread, 0 flat
    cooldown_left: int = 0


class IntradayPairsTrader:
    """
    Stateful live trader logic for a single pair.

    You provide the latest history DataFrame each call:
      index: timestamp
      columns: [asset_a, asset_b]
    """

    def __init__(
        self,
        asset_a: str,
        asset_b: str,
        cfg: Optional[IntradayConfig] = None,
        adapter: Optional[ZerodhaAdapter] = None,
    ) -> None:
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.cfg = cfg or IntradayConfig()
        self.adapter = adapter or ZerodhaAdapter(paper=True)
        self.state = TraderState()

    def on_new_history(self, history: pd.DataFrame) -> None:
        if len(history) < max(self.cfg.hedge_window_bars, self.cfg.zscore_window_bars) + 5:
            return

        px = history[[self.asset_a, self.asset_b]].dropna().copy()
        log_a = np.log(px[self.asset_a])
        log_b = np.log(px[self.asset_b])

        hr = rolling_ols_hedge_ratio(log_a, log_b, window=self.cfg.hedge_window_bars)
        spread = build_spread(log_a, log_b, hr["alpha"], hr["beta"])
        z = rolling_zscore(spread, self.cfg.zscore_window_bars)
        sig = raw_signal_from_z(
            z,
            type("TmpCfg", (), {"z_entry": self.cfg.z_entry, "z_exit": self.cfg.z_exit, "z_stop": self.cfg.z_stop}),
        )

        corr = rolling_corr(px[self.asset_a], px[self.asset_b], self.cfg.hedge_window_bars)
        vol_spike = volatility_spike_ratio(spread, self.cfg.hedge_window_bars)
        beta_jump = hr["beta"].diff().abs()

        last_ts = px.index[-1]
        last_z = float(z.iloc[-1]) if np.isfinite(z.iloc[-1]) else np.nan
        last_sig = int(sig.iloc[-1]) if np.isfinite(sig.iloc[-1]) else 0
        last_beta = float(hr["beta"].iloc[-1]) if np.isfinite(hr["beta"].iloc[-1]) else 1.0

        last_health = health_score(
            corr_value=float(corr.iloc[-1]) if np.isfinite(corr.iloc[-1]) else np.nan,
            beta_jump=float(beta_jump.iloc[-1]) if np.isfinite(beta_jump.iloc[-1]) else np.nan,
            vol_spike=float(vol_spike.iloc[-1]) if np.isfinite(vol_spike.iloc[-1]) else np.nan,
            min_corr=self.cfg.min_rolling_corr,
            max_beta_jump=self.cfg.max_beta_jump,
            max_vol_spike_ratio=self.cfg.max_vol_spike_ratio,
        )

        if self.state.cooldown_left > 0:
            self.state.cooldown_left -= 1
            return

        if self.state.position != 0 and np.isfinite(last_z) and abs(last_z) >= self.cfg.z_stop:
            self._exit_pair(last_ts)
            self.state.cooldown_left = self.cfg.cooldown_bars_after_stop
            return

        if last_health < self.cfg.health_pause_threshold:
            if self.state.position != 0:
                self._exit_pair(last_ts)
            return

        if self.state.position == 0 and last_sig != 0:
            self._enter_pair(last_sig, last_beta, float(px[self.asset_a].iloc[-1]), float(px[self.asset_b].iloc[-1]))
            return

        if self.state.position != 0 and last_sig == 0:
            self._exit_pair(last_ts)

    def _enter_pair(self, direction: int, beta: float, price_a: float, price_b: float) -> None:
        pair_capital = min(self.cfg.max_capital_per_pair, self.cfg.capital * self.cfg.max_gross_leverage / 2.0)
        size = compute_pair_order_size(price_a=price_a, price_b=price_b, beta=beta, pair_capital=pair_capital)

        if direction == 1:
            # Long spread: buy A, sell B
            orders = [
                OrderIntent(symbol=self.asset_a, side="BUY", quantity=size.qty_a),
                OrderIntent(symbol=self.asset_b, side="SELL", quantity=size.qty_b),
            ]
        else:
            # Short spread: sell A, buy B
            orders = [
                OrderIntent(symbol=self.asset_a, side="SELL", quantity=size.qty_a),
                OrderIntent(symbol=self.asset_b, side="BUY", quantity=size.qty_b),
            ]

        for o in orders:
            self.adapter.place_order(o)
        self.state.position = direction

    def _exit_pair(self, _ts: pd.Timestamp) -> None:
        # In a production engine, you should fetch actual open quantities from broker positions.
        # Here we place placeholder 1-share flattening orders as control-flow example.
        if self.state.position == 1:
            self.adapter.place_order(OrderIntent(symbol=self.asset_a, side="SELL", quantity=1))
            self.adapter.place_order(OrderIntent(symbol=self.asset_b, side="BUY", quantity=1))
        elif self.state.position == -1:
            self.adapter.place_order(OrderIntent(symbol=self.asset_a, side="BUY", quantity=1))
            self.adapter.place_order(OrderIntent(symbol=self.asset_b, side="SELL", quantity=1))
        self.state.position = 0
