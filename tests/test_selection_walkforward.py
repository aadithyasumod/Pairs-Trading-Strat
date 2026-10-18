import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from pairs_trading.config import StrategyConfig
from pairs_trading.data import generate_synthetic_universe
from pairs_trading.pair_selection import correlation_candidates, find_candidate_pairs, passes_screen, screen_pair
from pairs_trading.walk_forward import walk_forward


class PairSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.universe = generate_synthetic_universe(n_days=1000, seed=7, end="2026-01-01").prices

    def test_selects_cointegrated_pairs_and_rejects_decoys(self) -> None:
        pairs = find_candidate_pairs(self.universe, StrategyConfig(min_correlation=0.5))
        found = set(zip(pairs["asset_a"], pairs["asset_b"]))
        self.assertEqual(found, {("P1_A", "P1_B"), ("P2_A", "P2_B"), ("P3_A", "P3_B")})

    def test_high_correlation_is_not_cointegration(self) -> None:
        for k in (1, 2):
            row = screen_pair(self.universe[f"D{k}_A"], self.universe[f"D{k}_B"])
            self.assertGreater(row["corr"], 0.7)
            self.assertGreater(row["eg_pvalue"], 0.05)

    def test_screen_recovers_hedge_ratio_and_half_life(self) -> None:
        row = screen_pair(self.universe["P2_A"], self.universe["P2_B"])
        self.assertLess(row["eg_pvalue"], 0.01)
        self.assertGreater(row["beta"], 0.0)
        self.assertAlmostEqual(row["half_life"], 10.0, delta=5.0)

    def test_half_life_bounds(self) -> None:
        cfg = StrategyConfig(min_half_life=1.0, max_half_life=60.0)
        base = {"eg_pvalue": 0.01, "beta": 1.0}
        self.assertTrue(passes_screen({**base, "half_life": 8.0}, cfg))
        self.assertFalse(passes_screen({**base, "half_life": 150.0}, cfg))
        self.assertFalse(passes_screen({**base, "half_life": 0.5}, cfg))
        self.assertFalse(passes_screen({**base, "half_life": float("nan")}, cfg))
        self.assertFalse(passes_screen({**base, "half_life": 8.0, "beta": -0.3}, cfg))

    def test_correlation_prefilter_uses_log_returns(self) -> None:
        # Prices below 1 have negative logs; pct_change of log prices flips the sign of their moves.
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2024-01-01", periods=300)
        cheap = pd.Series(0.5 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))), index=idx)
        prices = pd.DataFrame({"CHEAP": cheap, "TWIN": cheap * 400.0, "OTHER": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))})
        self.assertIn(("CHEAP", "TWIN"), correlation_candidates(prices, min_corr=0.9, top_neighbors=5, max_candidates=10))


class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.universe = generate_synthetic_universe(seed=8, end="2026-01-01").prices
        cls.cfg = StrategyConfig()
        cls.res = walk_forward(cls.universe, cls.cfg)

    def test_pnl_covers_only_trading_windows(self) -> None:
        self.assertEqual(self.res.pnl.index[0], self.universe.index[self.cfg.formation_days])
        self.assertEqual(self.res.pnl.index[-1], self.universe.index[-1])
        self.assertTrue(self.res.pnl.index.is_unique)

    def test_selection_ignores_the_future(self) -> None:
        cut = self.cfg.formation_days
        altered = self.universe.copy()
        altered.iloc[cut:] = altered.iloc[cut:].sample(frac=1.0, axis=1, random_state=0).to_numpy()
        other = walk_forward(altered, self.cfg)
        self.assertEqual(self.res.windows["pairs"].iloc[0], other.windows["pairs"].iloc[0])

    def test_trades_stay_inside_their_window(self) -> None:
        windows = self.res.windows.set_index("window")
        for t in self.res.trades.itertuples():
            w = windows.loc[t.window]
            self.assertGreater(t.entry_time, w.trading_start)
            self.assertLessEqual(t.exit_time, w.trading_end)

    def test_out_of_sample_edge_on_stable_pairs(self) -> None:
        selected = self.res.windows["pairs"].str.split(", ").explode()
        selected = selected[selected.str.len() > 0]
        self.assertGreater(selected.str.startswith("P").mean(), 0.9)
        self.assertGreater(self.res.metrics["sharpe"], 0.5)
        self.assertGreater(self.res.metrics["n_trades"], 30)

    def test_rejects_formation_shorter_than_warm_up(self) -> None:
        with self.assertRaises(ValueError):
            walk_forward(self.universe, replace(self.cfg, formation_days=100))


if __name__ == "__main__":
    unittest.main()
