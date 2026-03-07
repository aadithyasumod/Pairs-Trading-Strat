from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import load_price_csv
from .live_analysis import _analyze_pair, _build_candidates, _download_prices, _filter_symbols, _read_symbols


@dataclass
class MonitorConfig:
    # Data source: use either csv path OR symbols_file
    csv: str | None = None
    symbols_file: str | None = None
    symbol_column: str | None = "Symbol"
    start: str = "2020-01-01"
    end: str | None = None
    chunk_size: int = 100
    sleep_seconds: float = 0.5

    # Analysis settings
    min_history: int = 252
    min_corr: float = 0.65
    top_neighbors: int = 20
    max_candidates: int = 40000
    coint_p_threshold: float = 0.05
    top_print: int = 15

    # Alert thresholds
    alert_zscore: float = 2.0
    alert_confidence: float = 1.0
    health_min_for_alert: float = 0.4

    # Runtime
    loop_minutes: int = 0
    bell_alert: bool = True
    webhook_url: str | None = None

    # Outputs
    out_csv: str | None = "/Users/aadithyasumod/Documents/Playground/live_analysis_latest.csv"
    out_json: str | None = "/Users/aadithyasumod/Documents/Playground/live_analysis_summary.json"



def _load_config(path: str) -> MonitorConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = json.loads(p.read_text())
    return MonitorConfig(**data)



def _fetch_prices(cfg: MonitorConfig) -> pd.DataFrame:
    if cfg.csv:
        return load_price_csv(cfg.csv).prices
    if cfg.symbols_file:
        symbols = _read_symbols(cfg.symbols_file, cfg.symbol_column)
        return _download_prices(symbols, cfg.start, cfg.end, cfg.chunk_size, cfg.sleep_seconds)
    raise RuntimeError("Config must set either 'csv' or 'symbols_file'.")



def _strategy_health(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    avg_health = float(df["health"].mean()) if "health" in df else 0.0
    actionable_rate = float((df["status"] == "ACTIONABLE").mean()) if "status" in df else 0.0
    conf = float(np.clip(df["confidence"].mean() / 3.0, 0.0, 1.0)) if "confidence" in df else 0.0
    score = 0.5 * avg_health + 0.3 * actionable_rate + 0.2 * conf
    return float(np.clip(score, 0.0, 1.0))



def _send_webhook(url: str, payload: dict) -> None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=8).read()



def _run_once(cfg: MonitorConfig) -> tuple[pd.DataFrame, dict]:
    s_cfg = StrategyConfig(min_history=cfg.min_history, min_correlation=cfg.min_corr)

    prices = _fetch_prices(cfg)
    prices = _filter_symbols(prices, cfg.min_history)
    candidates = _build_candidates(prices, cfg.min_corr, cfg.top_neighbors, cfg.max_candidates)

    rows = []
    for a, b in candidates:
        row = _analyze_pair(prices, a, b, s_cfg)
        if row and row["eg_pvalue"] <= cfg.coint_p_threshold:
            rows.append(row)

    if not rows:
        raise RuntimeError("No cointegrated pairs found with current settings.")

    df = pd.DataFrame(rows)
    prio = {"ACTIONABLE": 0, "WATCHLIST": 1, "WATCH_REGIME_WEAK": 2, "NEAR_MEAN": 3, "NO_SIGNAL": 4, "EXTREME_RISK": 5}
    df["priority"] = df["status"].map(prio).fillna(99)
    df = df.sort_values(["priority", "confidence"], ascending=[True, False]).drop(columns=["priority"]).reset_index(drop=True)

    actionable = df[
        (df["status"] == "ACTIONABLE")
        & (df["health"] >= cfg.health_min_for_alert)
        & (df["confidence"] >= cfg.alert_confidence)
        & (df["zscore"].abs() >= cfg.alert_zscore)
    ]

    summary = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "symbols_analyzed": int(prices.shape[1]),
        "candidate_pairs": int(len(candidates)),
        "cointegrated_pairs": int(len(df)),
        "actionable_pairs": int((df["status"] == "ACTIONABLE").sum()),
        "alert_pairs": int(len(actionable)),
        "strategy_health_score": _strategy_health(df),
    }

    if len(df) > 0:
        top = df.iloc[0]
        summary["top_pair"] = f"{top['asset_a']}/{top['asset_b']}"
        summary["top_status"] = str(top["status"])
        summary["top_zscore"] = float(top["zscore"])
        summary["top_confidence"] = float(top["confidence"])

    return df, summary



def _alert_if_needed(cfg: MonitorConfig, df: pd.DataFrame, summary: dict) -> None:
    alert_pairs = df[
        (df["status"] == "ACTIONABLE")
        & (df["health"] >= cfg.health_min_for_alert)
        & (df["confidence"] >= cfg.alert_confidence)
        & (df["zscore"].abs() >= cfg.alert_zscore)
    ]

    if alert_pairs.empty:
        return

    top = alert_pairs.iloc[0]
    msg = (
        f"ALERT: {top['asset_a']}/{top['asset_b']} | {top['suggested_action']} | "
        f"z={top['zscore']:.2f}, health={top['health']:.2f}, conf={top['confidence']:.2f}"
    )
    print("\n" + msg)

    if cfg.bell_alert:
        print("\a", end="")

    webhook = cfg.webhook_url or os.getenv("PAIRS_ALERT_WEBHOOK_URL")
    if webhook:
        payload = {
            "text": msg,
            "summary": summary,
        }
        try:
            _send_webhook(webhook, payload)
            print("Webhook alert sent.")
        except Exception as exc:
            print(f"Webhook alert failed: {exc}")



def _print_report(df: pd.DataFrame, summary: dict, top_print: int) -> None:
    print("\nStrategy Summary")
    for k, v in summary.items():
        print(f"{k}: {v}")

    show = [
        "date",
        "asset_a",
        "asset_b",
        "status",
        "suggested_action",
        "zscore",
        "health",
        "eg_pvalue",
        "confidence",
    ]
    print("\nTop opportunities")
    print(df[show].head(top_print).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple live pairs monitor (analysis only)")
    p.add_argument(
        "--config",
        default="/Users/aadithyasumod/Documents/Playground/monitor_config.json",
        help="Path to monitor config JSON",
    )
    return p.parse_args()



def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    while True:
        df, summary = _run_once(cfg)
        _print_report(df, summary, cfg.top_print)

        if cfg.out_csv:
            Path(cfg.out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cfg.out_csv, index=False)

        if cfg.out_json:
            Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.out_json).write_text(json.dumps(summary, indent=2))

        _alert_if_needed(cfg, df, summary)

        if cfg.loop_minutes <= 0:
            break

        print(f"\nSleeping for {cfg.loop_minutes} minutes...")
        time.sleep(cfg.loop_minutes * 60)


if __name__ == "__main__":
    main()
