"""Smoke tests: every command-line entrypoint runs end to end on generated data."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pairs_trading.data import generate_synthetic_intraday, generate_synthetic_universe

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.daily = d / "daily.csv"
        generate_synthetic_universe(n_days=900, seed=7, end="2026-09-18").prices.reset_index().to_csv(cls.daily, index=False)
        cls.intraday = d / "intraday.csv"
        generate_synthetic_intraday(n_sessions=8, seed=3).reset_index().to_csv(cls.intraday, index=False)
        cls.config = d / "monitor.json"
        cls.config.write_text(
            json.dumps(
                {
                    "csv": str(cls.daily),
                    "min_corr": 0.5,
                    "max_data_age_days": None,
                    "bell_alert": False,
                    "out_csv": str(d / "latest.csv"),
                    "out_json": str(d / "summary.json"),
                    "results_dir": str(d / "results"),
                }
            )
        )
        cls.dir = d

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_module(self, module: str, *args: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-m", f"pairs_trading.{module}", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(proc.returncode, 0, f"{module} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
        return proc.stdout

    def test_main_synthetic(self) -> None:
        out = self.run_module("main")
        self.assertIn("Selected pair: HDFCBANK / ICICIBANK", out)

    def test_main_csv_explicit_pair(self) -> None:
        out = self.run_module("main", "--csv", str(self.daily), "--asset-a", "P2_A", "--asset-b", "P2_B")
        self.assertIn("Backtest summary", out)

    def test_batch_daily(self) -> None:
        out_csv = self.dir / "ranking.csv"
        self.run_module("batch_daily", "--csv", str(self.daily), "--out", str(out_csv))
        self.assertTrue(out_csv.exists())

    def test_universe_scan(self) -> None:
        self.run_module("universe_scan", "--csv", str(self.daily), "--min-corr", "0.5", "--max-backtests", "10")

    def test_walk_forward(self) -> None:
        out = self.run_module("walk_forward", "--csv", str(self.daily), "--top-pairs", "3")
        self.assertIn("Out-of-sample portfolio", out)

    def test_live_analysis(self) -> None:
        self.run_module("live_analysis", "--csv", str(self.daily), "--min-corr", "0.5")

    def test_simple_monitor(self) -> None:
        out = self.run_module("simple_monitor", "--config", str(self.config))
        self.assertIn("strategy_traffic_light", out)
        self.assertTrue((self.dir / "summary.json").exists())

    def test_run_intraday(self) -> None:
        out = self.run_module("run_intraday", "--csv", str(self.intraday), "--asset-a", "HDFCBANK", "--asset-b", "ICICIBANK")
        self.assertIn("Trades:", out)

    def test_batch_intraday(self) -> None:
        self.run_module("batch_intraday", "--csv", str(self.intraday), "--min-bars", "200")

    def test_replay_live(self) -> None:
        out = self.run_module("replay_live", "--csv", str(self.intraday), "--asset-a", "HDFCBANK", "--asset-b", "ICICIBANK")
        self.assertIn("flat at end: True", out)


if __name__ == "__main__":
    unittest.main()
