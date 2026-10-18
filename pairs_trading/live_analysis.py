from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .backtest import build_daily_model
from .config import StrategyConfig
from .data import load_price_csv
from .pair_selection import correlation_candidates, filter_symbols, passes_screen, screen_pair

Pair = Tuple[str, str]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live pairs analysis scanner (no execution)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Wide daily CSV with Date + symbols")
    src.add_argument("--symbols-file", help="TXT/CSV of NSE symbols for online download")

    p.add_argument("--symbol-column", default=None, help="Symbol column if symbols-file is CSV")
    p.add_argument("--start", default="2020-01-01", help="History start date if using --symbols-file")
    p.add_argument("--end", default=None, help="History end date if using --symbols-file")
    p.add_argument("--chunk-size", type=int, default=100, help="Download chunk size for yfinance")
    p.add_argument("--sleep-seconds", type=float, default=0.5, help="Pause between chunks")

    p.add_argument("--min-history", type=int, default=252)
    p.add_argument("--min-corr", type=float, default=0.65)
    p.add_argument("--top-neighbors", type=int, default=20)
    p.add_argument("--max-candidates", type=int, default=40000)
    p.add_argument("--coint-p-threshold", type=float, default=0.05)
    p.add_argument("--top", type=int, default=30, help="Top rows to print")
    p.add_argument("--out", default=None, help="Optional output CSV")
    return p.parse_args()



def _read_symbols(path: str, column: str | None) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Symbol file not found: {path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if column and column in df.columns:
            vals = df[column].dropna().astype(str).tolist()
        elif "Symbol" in df.columns:
            vals = df["Symbol"].dropna().astype(str).tolist()
        else:
            vals = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        vals = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    clean = []
    for v in vals:
        s = v.strip().upper().replace(".NS", "")
        if s:
            clean.append(s)

    return list(dict.fromkeys(clean))



def _to_yahoo(symbol: str) -> str:
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"



def _download_prices(symbols: List[str], start: str, end: str | None, chunk_size: int, sleep_seconds: float) -> pd.DataFrame:
    import yfinance as yf  # only needed when downloading, so CSV mode works offline

    yahoo_symbols = [_to_yahoo(s) for s in symbols]
    chunks = [yahoo_symbols[i : i + chunk_size] for i in range(0, len(yahoo_symbols), chunk_size)]

    frames = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Downloading chunk {i}/{len(chunks)} ({len(chunk)} symbols)")
        data = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            time.sleep(sleep_seconds)
            continue

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" not in set(data.columns.get_level_values(0)):
                time.sleep(sleep_seconds)
                continue
            px = data["Close"].copy()
        else:
            if "Close" not in data.columns:
                time.sleep(sleep_seconds)
                continue
            px = data[["Close"]].copy()
            if len(chunk) == 1:
                px.columns = [chunk[0]]

        frames.append(px)
        time.sleep(sleep_seconds)

    if not frames:
        raise RuntimeError("No data downloaded.")

    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.rename(columns={c: str(c).replace(".NS", "") for c in merged.columns})
    merged = merged.dropna(how="all").ffill().dropna(how="any")
    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "Date"
    return merged



def _filter_symbols(prices: pd.DataFrame, min_history: int) -> pd.DataFrame:
    return filter_symbols(prices, min_history)



def _build_candidates(prices: pd.DataFrame, min_corr: float, top_neighbors: int, max_candidates: int) -> List[Pair]:
    return correlation_candidates(prices, min_corr, top_neighbors, max_candidates)



def _status_from_signal(z: float, health: float, cfg: StrategyConfig, health_pause_threshold: float) -> Tuple[str, str]:
    if not np.isfinite(z):
        return "NO_SIGNAL", "none"

    if abs(z) >= cfg.z_stop:
        return "EXTREME_RISK", "avoid"

    if abs(z) >= cfg.z_entry:
        if health < health_pause_threshold:
            return "WATCH_REGIME_WEAK", "none"
        if z > 0:
            return "ACTIONABLE", "short_spread (short A, long B)"
        return "ACTIONABLE", "long_spread (long A, short B)"

    if abs(z) <= cfg.z_exit:
        return "NEAR_MEAN", "none"

    return "WATCHLIST", "none"



def _analyze_pair(prices: pd.DataFrame, a: str, b: str, cfg: StrategyConfig, require_screen: bool = False) -> dict:
    """Current signal for one pair.

    Cointegration, hedge beta and formation half-life are measured on the most recent
    `cfg.formation_days` bars (the same window the walk-forward backtest selects on). With
    `require_screen`, pairs that fail the screen return {} before the costlier signal model runs.
    """
    pair_px = prices[[a, b]].dropna()
    if len(pair_px) < cfg.min_history:
        return {}

    formation = pair_px.iloc[-cfg.formation_days :] if cfg.formation_days else pair_px
    screen = screen_pair(formation[a], formation[b])
    if screen is None:
        return {}
    if require_screen and not passes_screen(screen, cfg):
        return {}

    model = build_daily_model(pair_px[a], pair_px[b], cfg)
    spread_std = model["spread"].rolling(cfg.zscore_window).std(ddof=0)
    last = model.iloc[-1]

    def finite(value: float) -> float:
        return float(value) if np.isfinite(value) else np.nan

    last_z = finite(last["zscore"])
    last_spread_std = finite(spread_std.iloc[-1])
    health = float(last["health"])
    pval = screen["eg_pvalue"]

    status, suggested_action = _status_from_signal(last_z, health, cfg, health_pause_threshold=cfg.health_pause_threshold)

    confidence = abs(last_z) * max(0.0, health) * max(0.0, 1.0 - pval) if np.isfinite(last_z) else 0.0
    # Approximate expected pair return if spread mean-reverts from current z toward z_exit.
    expected_reversion_return = np.nan
    if np.isfinite(last_z) and np.isfinite(last_spread_std):
        z_gap = max(0.0, abs(last_z) - cfg.z_exit)
        expected_reversion_return = z_gap * max(0.0, last_spread_std)
        expected_reversion_return = float(np.clip(expected_reversion_return, 0.0, 0.25))

    return {
        "date": pair_px.index[-1],
        "asset_a": a,
        "asset_b": b,
        "corr": screen["corr"],
        "eg_pvalue": float(pval),
        "beta": finite(last["beta"]),
        "half_life": finite(last["half_life"]),
        "formation_half_life": screen["half_life"],
        "zscore": last_z,
        "spread_std": last_spread_std,
        "health": health,
        "status": status,
        "suggested_action": suggested_action,
        "confidence": float(confidence),
        "expected_reversion_return": expected_reversion_return,
    }



def main() -> None:
    args = parse_args()
    cfg = StrategyConfig(
        min_history=args.min_history, min_correlation=args.min_corr, coint_p_threshold=args.coint_p_threshold
    )

    if args.csv:
        prices = load_price_csv(args.csv).prices
    else:
        symbols = _read_symbols(args.symbols_file, args.symbol_column)
        prices = _download_prices(symbols, args.start, args.end, args.chunk_size, args.sleep_seconds)

    prices = _filter_symbols(prices, cfg.min_history)
    print(f"Symbols after quality filter: {prices.shape[1]}")

    candidates = _build_candidates(prices, args.min_corr, args.top_neighbors, args.max_candidates)
    print(f"Candidate pairs: {len(candidates)}")

    if not candidates:
        raise RuntimeError("No candidates found. Lower --min-corr.")

    rows = []
    for i, (a, b) in enumerate(candidates, 1):
        row = _analyze_pair(prices, a, b, cfg, require_screen=True)
        if row:
            rows.append(row)

        if i % 1000 == 0:
            print(f"Analyzed {i}/{len(candidates)} pairs, passed coint={len(rows)}")

    if not rows:
        raise RuntimeError("No cointegrated pairs found after analysis.")

    df = pd.DataFrame(rows)
    priority = {"ACTIONABLE": 0, "WATCHLIST": 1, "WATCH_REGIME_WEAK": 2, "NEAR_MEAN": 3, "NO_SIGNAL": 4, "EXTREME_RISK": 5}
    df["priority"] = df["status"].map(priority).fillna(99)
    df = df.sort_values(["priority", "confidence"], ascending=[True, False]).drop(columns=["priority"])

    display_cols = [
        "date",
        "asset_a",
        "asset_b",
        "status",
        "suggested_action",
        "zscore",
        "health",
        "eg_pvalue",
        "corr",
        "half_life",
        "confidence",
    ]

    print("\nTop opportunities:")
    print(df[display_cols].head(args.top).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved analysis: {args.out}")


if __name__ == "__main__":
    main()
