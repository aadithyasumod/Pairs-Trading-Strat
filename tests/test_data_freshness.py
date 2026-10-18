import unittest

import pandas as pd

from pairs_trading.data import require_fresh_price_data


class PriceDataFreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = pd.DataFrame(
            {"A": [100.0, 101.0], "B": [200.0, 199.0]},
            index=pd.to_datetime(["2026-09-17", "2026-09-18"]),
        )

    def test_returns_data_timestamp_and_age_when_within_limit(self) -> None:
        freshness = require_fresh_price_data(
            self.prices, max_age_days=7, as_of="2026-09-20"
        )

        self.assertEqual(freshness.last_observation, pd.Timestamp("2026-09-18"))
        self.assertEqual(freshness.age_days, 2)

    def test_rejects_a_stale_monitoring_panel(self) -> None:
        with self.assertRaisesRegex(ValueError, "Price data is stale for live monitoring"):
            require_fresh_price_data(
                self.prices, max_age_days=1, as_of="2026-09-20"
            )

    def test_none_limit_allows_historical_research(self) -> None:
        freshness = require_fresh_price_data(
            self.prices, max_age_days=None, as_of="2026-10-20"
        )

        self.assertEqual(freshness.age_days, 32)


if __name__ == "__main__":
    unittest.main()
