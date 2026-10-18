import math
import unittest

import numpy as np
import pandas as pd

from pairs_trading.data import generate_synthetic_prices, ou_path
from pairs_trading.hedge_ratio import rolling_ols_hedge_ratio
from pairs_trading.ou import estimate_half_life, half_life
from pairs_trading.signals import rolling_zscore, spread_model


def _loop_ols(log_a: pd.Series, log_b: pd.Series, window: int) -> pd.DataFrame:
    """Reference implementation: one least-squares fit per window."""
    betas = np.full(len(log_a), np.nan)
    alphas = np.full(len(log_a), np.nan)
    for i in range(window - 1, len(log_a)):
        y = log_a.iloc[i - window + 1 : i + 1].to_numpy()
        x = log_b.iloc[i - window + 1 : i + 1].to_numpy()
        coeff, *_ = np.linalg.lstsq(np.column_stack([np.ones(window), x]), y, rcond=None)
        alphas[i], betas[i] = coeff
    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=log_a.index)


def _loop_half_life(spread: pd.Series, window: int) -> pd.Series:
    """Reference implementation: explicit regression on the `window` values before t."""
    out = pd.Series(np.nan, index=spread.index)
    for i in range(window, len(spread)):
        s = spread.iloc[i - window : i]
        ds = s.diff().dropna().to_numpy()
        lag = s.shift(1).dropna().to_numpy()
        coeff, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(lag)), lag]), ds, rcond=None)
        b = coeff[1]
        if b < 0:
            phi = 1 + b
            out.iloc[i] = -math.log(2) / math.log(phi) if phi > 0 else math.log(2) / -b
    return out


class HedgeRatioTests(unittest.TestCase):
    def test_vectorised_rolling_ols_matches_least_squares_loop(self) -> None:
        px = generate_synthetic_prices(n_days=400, seed=3, end="2026-01-01").prices
        log_a, log_b = np.log(px["HDFCBANK"]), np.log(px["ICICIBANK"])
        fast = rolling_ols_hedge_ratio(log_a, log_b, window=60)
        slow = _loop_ols(log_a, log_b, window=60)
        pd.testing.assert_frame_equal(fast, slow, atol=1e-8, rtol=1e-6)

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            rolling_ols_hedge_ratio(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0]), window=2)


class HalfLifeTests(unittest.TestCase):
    def test_vectorised_rolling_half_life_matches_regression_loop(self) -> None:
        rng = np.random.default_rng(0)
        spread = pd.Series(ou_path(600, half_life=8.0, stationary_std=0.02, rng=rng))
        fast = estimate_half_life(spread, window=120)
        slow = _loop_half_life(spread, window=120)
        pd.testing.assert_series_equal(fast, slow, check_names=False, atol=1e-8, rtol=1e-6)

    def test_rolling_half_life_at_t_ignores_spread_at_t(self) -> None:
        rng = np.random.default_rng(1)
        spread = pd.Series(ou_path(300, half_life=8.0, stationary_std=0.02, rng=rng))
        base = estimate_half_life(spread, window=60)
        bumped = spread.copy()
        bumped.iloc[200] += 1.0
        self.assertEqual(base.iloc[200], estimate_half_life(bumped, window=60).iloc[200])

    def test_recovers_known_half_life(self) -> None:
        for true_hl in (5.0, 15.0):
            rng = np.random.default_rng(42)
            spread = pd.Series(ou_path(5000, half_life=true_hl, stationary_std=0.02, rng=rng))
            self.assertAlmostEqual(half_life(spread), true_hl, delta=0.2 * true_hl)

    def test_random_walk_has_no_meaningful_half_life(self) -> None:
        rng = np.random.default_rng(5)
        walk = pd.Series(np.cumsum(rng.normal(0, 0.01, 3000)))
        hl = half_life(walk)
        self.assertTrue(math.isnan(hl) or hl > 200, hl)


class SpreadModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.px = generate_synthetic_prices(n_days=600, seed=9, end="2026-01-01").prices

    def test_model_is_causal(self) -> None:
        a, b = self.px["HDFCBANK"], self.px["ICICIBANK"]
        base = spread_model(a, b, 120, 60, 120)
        shocked_a = a.copy()
        shocked_a.iloc[400:] *= 1.25  # rewrite the future
        shocked = spread_model(shocked_a, b, 120, 60, 120)
        pd.testing.assert_frame_equal(base.iloc[:400], shocked.iloc[:400])

    def test_hedge_ratio_at_t_is_fitted_before_t(self) -> None:
        a, b = self.px["HDFCBANK"], self.px["ICICIBANK"]
        base = spread_model(a, b, 120, 60, 120)
        bumped_a = a.copy()
        bumped_a.iloc[300] *= 1.05
        bumped = spread_model(bumped_a, b, 120, 60, 120)
        self.assertEqual(base["beta"].iloc[300], bumped["beta"].iloc[300])
        self.assertNotEqual(base["spread"].iloc[300], bumped["spread"].iloc[300])

    def test_rolling_beta_tracks_true_hedge_ratio(self) -> None:
        px = generate_synthetic_prices(n_days=1500, seed=1, beta=0.9, end="2026-01-01").prices
        model = spread_model(px["HDFCBANK"], px["ICICIBANK"], 250, 60, 120)
        self.assertAlmostEqual(model["beta"].median(), 0.9, delta=0.15)

    def test_zscore_of_flat_series_is_nan_not_infinite(self) -> None:
        z = rolling_zscore(pd.Series([1.0] * 10), window=5)
        self.assertFalse(np.isinf(z).any())
        self.assertTrue(z.iloc[4:].isna().all())

    def test_rejects_non_positive_prices(self) -> None:
        with self.assertRaises(ValueError):
            spread_model(pd.Series([1.0, 0.0]), pd.Series([1.0, 1.0]), 2, 2, 3)


if __name__ == "__main__":
    unittest.main()
