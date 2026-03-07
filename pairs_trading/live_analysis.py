from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint

from .config import StrategyConfig
from .data import load_price_csv
from .hedge_ratio import rolling_ols_hedge_ratio
from .ou import estimate_half_life
from .regime import health_score, rolling_corr, volatility_spike_ratio
from .signals import build_spread, rolling_zscore

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
    keep = []
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) >= min_history and (s > 0).all():
            keep.append(c)
    if len(keep) < 2:
        raise RuntimeError("Not enough symbols after data-quality filter.")
    return prices[keep].copy()



def _build_candidates(prices: pd.DataFrame, min_corr: float, top_neighbors: int, max_candidates: int) -> List[Pair]:
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    corr = rets.corr(min_periods=60)

    pairs = set()
    for s in corr.columns:
        row = corr.loc[s].drop(labels=[s]).dropna()
        row = row[row >= min_corr].sort_values(ascending=False).head(top_neighbors)
        for t in row.index:
            a, b = sorted((s, t))
            pairs.add((a, b))
            if len(pairs) >= max_candidates:
                break
        if len(pairs) >= max_candidates:
            break

    return sorted(pairs)



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



def _analyze_pair(prices: pd.DataFrame, a: str, b: str, cfg: StrategyConfig) -> dict:
    pair_px = prices[[a, b]].dropna()
    if len(pair_px) < cfg.min_history:
        return {}

    corr_val = pair_px[a].pct_change().corr(pair_px[b].pct_change())
    if not np.isfinite(corr_val):
        return {}

    log_a = np.log(pair_px[a])
    log_b = np.log(pair_px[b])

    try:
        _, pval, _ = coint(log_a, log_b)
    except Exception:
        return {}

    hr = rolling_ols_hedge_ratio(log_a, log_b, window=cfg.hedge_window)
    spread = build_spread(log_a, log_b, hr["alpha"], hr["beta"])
    z = rolling_zscore(spread, cfg.zscore_window)
    hl = estimate_half_life(spread, cfg.ou_window)

    rcorr = rolling_corr(pair_px[a], pair_px[b], cfg.hedge_window)
    beta_jump = hr["beta"].diff().abs()
    vspike = volatility_spike_ratio(spread, cfg.hedge_window)

    last = pair_px.index[-1]
    last_z = float(z.loc[last]) if np.isfinite(z.loc[last]) else np.nan
    last_hl = float(hl.loc[last]) if np.isfinite(hl.loc[last]) else np.nan
    last_beta = float(hr["beta"].loc[last]) if np.isfinite(hr["beta"].loc[last]) else np.nan

    health = health_score(
        corr_value=float(rcorr.loc[last]) if np.isfinite(rcorr.loc[last]) else np.nan,
        beta_jump=float(beta_jump.loc[last]) if np.isfinite(beta_jump.loc[last]) else np.nan,
        vol_spike=float(vspike.loc[last]) if np.isfinite(vspike.loc[last]) else np.nan,
        min_corr=0.45,
        max_beta_jump=0.08,
        max_vol_spike_ratio=2.0,
    )

    status, suggested_action = _status_from_signal(last_z, health, cfg, health_pause_threshold=0.4)

    confidence = abs(last_z) * max(0.0, health) * max(0.0, 1.0 - pval)

    return {
        "date": last,
        "asset_a": a,
        "asset_b": b,
        "corr": float(corr_val),
        "eg_pvalue": float(pval),
        "beta": last_beta,
        "half_life": last_hl,
        "zscore": last_z,
        "health": float(health),
        "status": status,
        "suggested_action": suggested_action,
        "confidence": float(confidence),
    }



def main() -> None:
    args = parse_args()
    cfg = StrategyConfig(min_history=args.min_history, min_correlation=args.min_corr)

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
        row = _analyze_pair(prices, a, b, cfg)
        if row and row["eg_pvalue"] <= args.coint_p_threshold:
            rows.append(row)

        if i % 1000 == 0:
            print(f"Analyzed {i}/{len(candidates)} pairs, passed coint={len(rows)}")

    if not rows:
        raise RuntimeError("No cointegrated pairs found after analysis.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["status", "confidence", "health", "eg_pvalue"], ascending=[True, False, False, True])

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
