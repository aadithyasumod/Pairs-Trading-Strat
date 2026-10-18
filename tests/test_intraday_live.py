import datetime as dt
import unittest
from dataclasses import replace

import pandas as pd

from pairs_trading.data import generate_synthetic_intraday
from pairs_trading.intraday_config import IntradayConfig
from pairs_trading.intraday_engine import backtest_intraday_pair
from pairs_trading.live_trader import IntradayPairsTrader
from pairs_trading.zerodha_adapter import OrderIntent, ZerodhaAdapter

# Shorter windows keep the bar-by-bar replay fast; the rules are the same.
FAST_CFG = replace(IntradayConfig(), hedge_window_bars=60, zscore_window_bars=30, ou_window_bars=60)


class RecordingAdapter(ZerodhaAdapter):
    def __init__(self) -> None:
        super().__init__(paper=True)
        self.orders: list[OrderIntent] = []

    def place_order(self, order: OrderIntent) -> str:
        if order.quantity <= 0:
            raise ValueError("non-positive quantity")
        self.orders.append(order)
        return "paper"


class IntradaySessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.px = generate_synthetic_intraday(n_sessions=15, seed=21)
        cls.res = backtest_intraday_pair(cls.px["HDFCBANK"], cls.px["ICICIBANK"], FAST_CFG)

    def test_produces_trades(self) -> None:
        self.assertGreater(self.res.n_trades, 5)

    def test_never_holds_overnight(self) -> None:
        eod = self.res.positions.groupby(self.res.positions.index.date).last()
        self.assertTrue((eod == 0).all())
        entry_days = self.res.trades["entry_time"].dt.date
        exit_days = self.res.trades["exit_time"].dt.date
        self.assertTrue((entry_days == exit_days).all())

    def test_flat_from_square_off_time(self) -> None:
        late = self.res.positions[self.res.positions.index.time >= dt.time(15, 20)]
        self.assertTrue((late == 0).all())

    def test_entries_only_inside_entry_window(self) -> None:
        decided = self.res.signals[self.res.signals["reason"] == "entry"]["timestamp"]
        times = decided.dt.time
        self.assertTrue(((times >= dt.time(9, 30)) & (times < dt.time(15, 0))).all())

    def test_costs_fully_charged(self) -> None:
        free = backtest_intraday_pair(
            self.px["HDFCBANK"], self.px["ICICIBANK"], replace(FAST_CFG, fee_bps_per_leg=0, slippage_bps_per_leg=0)
        )
        self.assertAlmostEqual(
            free.metrics["total_pnl"] - self.res.metrics["total_pnl"], self.res.metrics["total_costs"], delta=1e-6
        )

    def test_health_exit_uses_lower_threshold_than_entry(self) -> None:
        cfg = IntradayConfig()
        self.assertLess(cfg.health_exit_threshold, cfg.health_pause_threshold)


class LiveTraderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.px = generate_synthetic_intraday(n_sessions=8, seed=21)
        cls.adapter = RecordingAdapter()
        cls.trader = IntradayPairsTrader("HDFCBANK", "ICICIBANK", FAST_CFG, cls.adapter)
        for i in range(len(cls.px)):
            cls.trader.on_new_history(cls.px.iloc[: i + 1])
        cls.backtest = backtest_intraday_pair(cls.px["HDFCBANK"], cls.px["ICICIBANK"], FAST_CFG)

    def test_replay_makes_the_same_decisions_as_the_backtest(self) -> None:
        live = pd.DataFrame(self.trader.decisions)[["timestamp", "target", "reason"]]
        bt = self.backtest.signals[["timestamp", "target", "reason"]]
        self.assertGreater(len(bt), 0)
        pd.testing.assert_frame_equal(live.reset_index(drop=True), bt.reset_index(drop=True))

    def test_exits_flatten_exactly_what_was_opened(self) -> None:
        net: dict[str, int] = {}
        for o in self.adapter.orders:
            net[o.symbol] = net.get(o.symbol, 0) + (o.quantity if o.side == "BUY" else -o.quantity)
        self.assertEqual(net, {"HDFCBANK": 0, "ICICIBANK": 0})
        self.assertTrue(self.trader.is_flat())
        # Every order is a real leg size, never the old 1-share placeholder.
        self.assertGreater(min(o.quantity for o in self.adapter.orders), 1)

    def test_each_bar_is_processed_once(self) -> None:
        trader = IntradayPairsTrader("HDFCBANK", "ICICIBANK", FAST_CFG, RecordingAdapter())
        history = self.px.iloc[:300]
        self.assertIsNotNone(trader.on_new_history(history))
        self.assertIsNone(trader.on_new_history(history))


class AdapterTests(unittest.TestCase):
    def test_rejects_invalid_orders(self) -> None:
        adapter = ZerodhaAdapter(paper=True)
        with self.assertRaises(ValueError):
            adapter.place_order(OrderIntent(symbol="SBIN", side="BUY", quantity=0))
        with self.assertRaises(ValueError):
            adapter.place_order(OrderIntent(symbol="SBIN", side="HOLD", quantity=5))


if __name__ == "__main__":
    unittest.main()
