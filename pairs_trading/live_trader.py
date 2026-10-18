from __future__ import annotations

from typing import Optional

import pandas as pd

from .intraday_config import IntradayConfig
from .intraday_engine import build_intraday_model, intraday_gates, intraday_rules, pair_capital
from .position_sizing import compute_pair_order_size
from .signals import FLAT, LONG_SPREAD, PairStateMachine
from .zerodha_adapter import OrderIntent, ZerodhaAdapter


class IntradayPairsTrader:
    """
    Stateful live trader logic for a single pair.

    It drives the same PairStateMachine as the intraday backtest, so a bar-by-bar replay makes the
    same entry/exit decisions. It tracks the signed share quantities it opened and flattens exactly
    those on exit.

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
        self.machine = PairStateMachine(
            intraday_rules(self.cfg), max_hold_cap=self.cfg.max_holding_bars_cap, default_hold=30
        )
        self.qty_a = 0  # signed shares currently held
        self.qty_b = 0
        self.last_bar: pd.Timestamp | None = None
        self.decisions: list[dict] = []

    @property
    def position(self) -> int:
        return self.machine.direction

    def lookback_bars(self) -> int:
        """Bars needed for every rolling statistic at the latest bar to be fully warmed up."""
        cfg = self.cfg
        return 2 * cfg.hedge_window_bars + max(cfg.zscore_window_bars, cfg.ou_window_bars) + 5

    def on_new_history(self, history: pd.DataFrame) -> tuple[int, str] | None:
        """Process the newest bar in `history` once. Returns (target_position, reason)."""
        px = history[[self.asset_a, self.asset_b]].dropna()
        if px.empty:
            return None
        ts = px.index[-1]
        if self.last_bar is not None and ts <= self.last_bar:
            return None
        self.last_bar = ts

        px = px.iloc[-self.lookback_bars() :]
        model = build_intraday_model(px[self.asset_a], px[self.asset_b], self.cfg)
        can_enter, regime_ok, force_exit = intraday_gates(px.index, model["health"], self.cfg, detect_session_end=False)
        last = model.iloc[-1]

        target, reason = self.machine.on_bar(
            float(last["zscore"]),
            float(last["half_life"]),
            float(last["beta"]),
            can_enter=bool(can_enter.iloc[-1]),
            regime_ok=bool(regime_ok.iloc[-1]),
            force_reason=force_exit.iloc[-1],
        )

        if reason == "entry":
            self._enter_pair(target, float(last["beta"]), float(px[self.asset_a].iloc[-1]), float(px[self.asset_b].iloc[-1]))
        elif reason.startswith("exit"):
            self._exit_pair()

        if reason not in ("hold", "flat"):
            self.decisions.append({"timestamp": ts, "target": target, "reason": reason, "zscore": float(last["zscore"])})
        return target, reason

    def _enter_pair(self, direction: int, beta: float, price_a: float, price_b: float) -> None:
        size = compute_pair_order_size(price_a=price_a, price_b=price_b, beta=beta, pair_capital=pair_capital(self.cfg))
        if size.qty_a == 0 or size.qty_b == 0:
            return  # capital too small for one share of a leg

        # Long spread: buy A, sell B. Short spread: sell A, buy B.
        side_a, side_b = ("BUY", "SELL") if direction == LONG_SPREAD else ("SELL", "BUY")
        self.adapter.place_order(OrderIntent(symbol=self.asset_a, side=side_a, quantity=size.qty_a))
        self.adapter.place_order(OrderIntent(symbol=self.asset_b, side=side_b, quantity=size.qty_b))
        self.qty_a = direction * size.qty_a
        self.qty_b = -direction * size.qty_b

    def _exit_pair(self) -> None:
        # Flatten the quantities this trader opened. A production engine should also reconcile
        # these against broker positions (partial fills, rejections, manual interventions).
        for symbol, qty in ((self.asset_a, self.qty_a), (self.asset_b, self.qty_b)):
            if qty > 0:
                self.adapter.place_order(OrderIntent(symbol=symbol, side="SELL", quantity=qty))
            elif qty < 0:
                self.adapter.place_order(OrderIntent(symbol=symbol, side="BUY", quantity=-qty))
        self.qty_a = 0
        self.qty_b = 0

    def is_flat(self) -> bool:
        return self.machine.direction == FLAT and self.qty_a == 0 and self.qty_b == 0
