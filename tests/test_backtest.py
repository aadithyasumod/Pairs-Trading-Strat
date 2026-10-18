import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from pairs_trading.backtest import backtest_pair, simulate_pair
from pairs_trading.config import StrategyConfig
from pairs_trading.data import generate_synthetic_prices
from pairs_trading.position_sizing import compute_pair_order_size
from pairs_trading.signals import SignalRules

RULES = SignalRules(z_entry=2.0, z_exit=0.5, z_stop=3.5)


def scripted_pair(z_path, sigma=0.01):
    """B is flat at 100 and log(A/B) = z * sigma, so the z path fully determines PnL."""
    z = pd.Series(z_path, dtype=float)
    pb = pd.Series(100.0, index=z.index)
    pa = 100.0 * np.exp(z * sigma)
    model = pd.DataFrame({"zscore": z, "beta": 1.0, "half_life": 5.0})
    return pa, pb, model


def run_scripted(z_path, **kw):
    pa, pb, model = scripted_pair(z_path)
    params = dict(notional=100_000, cost_bps_per_leg=0.0, max_hold_cap=20, execution_lag=0)
    params.update(kw)
    return pa, pb, simulate_pair(pa, pb, model, RULES, **params)


class ScriptedAccountingTests(unittest.TestCase):
    Z = [0, 0, -2.2, -1.5, -0.8, -0.3, 0, 0, 2.4, 1.0, 0.2, 0.0]

    def test_pnl_matches_hand_calculation(self) -> None:
        pa, _, res = run_scripted(self.Z)
        self.assertEqual(res.trades["exit_reason"].tolist(), ["exit_mean_reversion"] * 2)
        q1, q2 = res.trades["qty_a"].tolist()
        self.assertEqual((q1, q2), (int(100_000 / pa[2]), -int(100_000 / pa[8])))
        expected = q1 * (pa[5] - pa[2]) + q2 * (pa[10] - pa[8])  # the B leg never moves
        self.assertAlmostEqual(res.metrics["total_pnl"], expected, places=6)

    def test_trade_ledger_reconciles_with_equity(self) -> None:
        _, _, res = run_scripted(self.Z, cost_bps_per_leg=8.0)
        self.assertAlmostEqual(res.trades["net_pnl"].sum(), res.pnl.sum(), places=6)
        self.assertAlmostEqual(res.equity_curve.iloc[-1], res.pnl.sum(), places=6)

    def test_costs_charged_per_leg_on_every_fill(self) -> None:
        pa, pb, free = run_scripted(self.Z)
        _, _, costly = run_scripted(self.Z, cost_bps_per_leg=10.0)
        expected_costs = 0.0
        for t in costly.trades.itertuples():
            entry = abs(t.qty_a) * t.entry_price_a + abs(t.qty_b) * t.entry_price_b
            exit_ = abs(t.qty_a) * t.exit_price_a + abs(t.qty_b) * t.exit_price_b
            expected_costs += (entry + exit_) * 10.0 / 10_000
        self.assertAlmostEqual(free.metrics["total_pnl"] - costly.metrics["total_pnl"], expected_costs, places=6)
        self.assertAlmostEqual(costly.metrics["total_costs"], expected_costs, places=6)

    def test_signal_fills_on_the_next_bar_with_default_lag(self) -> None:
        pa, pb, model = scripted_pair(self.Z)
        res = simulate_pair(pa, pb, model, RULES, notional=100_000, cost_bps_per_leg=0.0, max_hold_cap=20)
        entries = res.signals[res.signals["reason"] == "entry"]["timestamp"].tolist()
        self.assertEqual(entries, [2, 8])
        self.assertEqual(res.trades["entry_time"].tolist(), [3, 9])
        # The entry bar's own move (the one that triggered the signal) is not credited.
        self.assertEqual(res.pnl.iloc[2], 0.0)

    def test_open_position_is_closed_at_end_of_data(self) -> None:
        _, _, res = run_scripted([0, 0, -2.5, -2.0, -1.8])
        self.assertEqual(res.trades["exit_reason"].tolist(), ["end_of_data"])
        self.assertEqual(res.trades["exit_time"].iloc[0], 4)

    def test_stop_then_cooldown(self) -> None:
        rules = SignalRules(2.0, 0.5, 3.5, cooldown_bars=2)
        pa, pb, model = scripted_pair([0, -2.5, -3.7, -2.5, -2.5, -2.5, -0.1])
        res = simulate_pair(pa, pb, model, rules, notional=100_000, cost_bps_per_leg=0, max_hold_cap=20, execution_lag=0)
        self.assertEqual(res.trades["exit_reason"].tolist(), ["exit_stop", "exit_mean_reversion"])
        self.assertEqual(res.trades["entry_time"].tolist(), [1, 5])  # bars 3 and 4 sit out

    def test_forced_exit_fills_immediately(self) -> None:
        pa, pb, model = scripted_pair([0, -2.5, -2.0, -1.5, -1.0])
        force = pd.Series([None, None, None, "exit_square_off", None], dtype=object)
        res = simulate_pair(
            pa, pb, model, RULES, notional=100_000, cost_bps_per_leg=0, max_hold_cap=20, execution_lag=1, force_exit=force
        )
        self.assertEqual(res.trades["exit_reason"].tolist(), ["exit_square_off"])
        self.assertEqual(res.trades["exit_time"].tolist(), [3])


class DailyBacktestTests(unittest.TestCase):
    cfg = StrategyConfig()

    def test_no_look_ahead(self) -> None:
        px = generate_synthetic_prices(n_days=900, seed=4, end="2026-01-01").prices
        a, b = px["HDFCBANK"], px["ICICIBANK"]
        base = backtest_pair(a, b, self.cfg)
        cut = 600
        future = a.copy()
        future.iloc[cut + 1 :] *= np.exp(np.random.default_rng(0).normal(0, 0.05, len(a) - cut - 1))
        other = backtest_pair(future, b, self.cfg)
        pd.testing.assert_series_equal(base.pnl.iloc[: cut + 1], other.pnl.iloc[: cut + 1])
        pd.testing.assert_series_equal(base.positions.iloc[: cut + 1], other.positions.iloc[: cut + 1])

    def test_cointegrated_pair_is_profitable_and_random_walks_are_not(self) -> None:
        coint_pnl, random_pnl = [], []
        for seed in range(8):
            px = generate_synthetic_prices(n_days=1500, seed=seed, end="2026-01-01").prices
            coint_pnl.append(backtest_pair(px["HDFCBANK"], px["ICICIBANK"], self.cfg).metrics["total_pnl"])
            random_pnl.append(backtest_pair(px["AXISBANK"], px["SBIN"], self.cfg).metrics["total_pnl"])
        self.assertTrue(all(p > 0 for p in coint_pnl), coint_pnl)
        self.assertLess(np.median(random_pnl), min(coint_pnl))

    def test_most_exits_are_mean_reversion_on_a_cointegrated_pair(self) -> None:
        px = generate_synthetic_prices(n_days=1500, seed=42, end="2026-01-01").prices
        res = backtest_pair(px["HDFCBANK"], px["ICICIBANK"], self.cfg)
        share = (res.trades["exit_reason"] == "exit_mean_reversion").mean()
        self.assertGreater(share, 0.5)
        self.assertGreater(res.metrics["win_rate"], 0.5)

    def test_trade_window_limits_trading_and_metrics(self) -> None:
        px = generate_synthetic_prices(n_days=900, seed=2, end="2026-01-01").prices
        start, end = px.index[500], px.index[750]
        res = backtest_pair(px["HDFCBANK"], px["ICICIBANK"], self.cfg, trade_start=start, trade_end=end)
        self.assertEqual(res.pnl.index[0], start)
        self.assertEqual(res.pnl.index[-1], end)
        if len(res.trades):
            self.assertTrue((res.trades["entry_time"] > start).all())
            self.assertTrue((res.trades["exit_time"] <= end).all())
        self.assertEqual(res.positions.iloc[-1], 0)

    def test_higher_costs_reduce_pnl(self) -> None:
        px = generate_synthetic_prices(n_days=1200, seed=6, end="2026-01-01").prices
        cheap = backtest_pair(px["HDFCBANK"], px["ICICIBANK"], replace(self.cfg, fee_bps=0, slippage_bps=0))
        dear = backtest_pair(px["HDFCBANK"], px["ICICIBANK"], replace(self.cfg, fee_bps=20, slippage_bps=10))
        self.assertGreater(cheap.metrics["total_pnl"], dear.metrics["total_pnl"])

    def test_needs_enough_history(self) -> None:
        px = generate_synthetic_prices(n_days=100, seed=0, end="2026-01-01").prices
        with self.assertRaises(ValueError):
            backtest_pair(px["HDFCBANK"], px["ICICIBANK"], self.cfg)


class PositionSizingTests(unittest.TestCase):
    def test_legs_are_beta_weighted_whole_shares(self) -> None:
        size = compute_pair_order_size(price_a=1700.0, price_b=1300.0, beta=0.9, pair_capital=1_000_000)
        self.assertEqual(size.qty_a, 588)
        self.assertEqual(size.qty_b, 692)
        self.assertAlmostEqual(size.notional_b / size.notional_a, 0.9, delta=0.01)

    def test_refuses_unusable_hedge_ratios(self) -> None:
        for beta in (-0.5, 0.0, float("nan"), 12.0):
            with self.assertRaises(ValueError):
                compute_pair_order_size(100.0, 100.0, beta, 10_000)

    def test_zero_quantity_when_capital_too_small(self) -> None:
        size = compute_pair_order_size(price_a=50_000.0, price_b=100.0, beta=1.0, pair_capital=10_000)
        self.assertEqual(size.qty_a, 0)


if __name__ == "__main__":
    unittest.main()
