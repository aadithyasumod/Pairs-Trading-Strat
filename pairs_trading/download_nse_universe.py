from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf



def _to_yahoo_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s:
        return s
    return s if s.endswith('.NS') else f"{s}.NS"



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
        elif len(df.columns) >= 1:
            vals = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            vals = []
    else:
        vals = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    clean = []
    for v in vals:
        s = v.strip().upper().replace(".NS", "")
        if s:
            clean.append(s)

    # Preserve order, remove duplicates
    return list(dict.fromkeys(clean))



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily NSE universe prices in chunks")
    p.add_argument("--symbols-file", required=True, help="CSV/TXT containing NSE symbols")
    p.add_argument("--symbol-column", default=None, help="CSV column name holding symbols")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--chunk-size", type=int, default=100, help="Symbols per yfinance request")
    p.add_argument("--sleep-seconds", type=float, default=0.6, help="Pause between chunks")
    p.add_argument(
        "--out",
        default="/Users/aadithyasumod/Documents/Playground/daily_prices_nse_universe.csv",
        help="Output CSV path",
    )
    return p.parse_args()



def _extract_close(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = set(data.columns.get_level_values(0))
        if "Close" in lvl0:
            px = data["Close"].copy()
        elif "Adj Close" in lvl0:
            px = data["Adj Close"].copy()
        else:
            return pd.DataFrame()
    else:
        # Single symbol case
        if "Close" in data.columns:
            px = data[["Close"]].copy()
            px.columns = ["SINGLE"]
        elif "Adj Close" in data.columns:
            px = data[["Adj Close"]].copy()
            px.columns = ["SINGLE"]
        else:
            return pd.DataFrame()

    return px



def main() -> None:
    args = parse_args()
    symbols = _read_symbols(args.symbols_file, args.symbol_column)
    if len(symbols) < 2:
        raise RuntimeError("Need at least 2 symbols in symbols file.")

    yahoo_symbols = [_to_yahoo_symbol(s) for s in symbols]
    chunks = [yahoo_symbols[i : i + args.chunk_size] for i in range(0, len(yahoo_symbols), args.chunk_size)]

    all_px = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Downloading chunk {i}/{len(chunks)} with {len(chunk)} symbols...")
        try:
            data = yf.download(
                chunk,
                start=args.start,
                end=args.end,
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
        except Exception as exc:
            print(f"Chunk failed: {exc}")
            time.sleep(args.sleep_seconds)
            continue

        px = _extract_close(data)
        if px.empty:
            time.sleep(args.sleep_seconds)
            continue

        if "SINGLE" in px.columns and len(chunk) == 1:
            px.columns = [chunk[0]]

        all_px.append(px)
        time.sleep(args.sleep_seconds)

    if not all_px:
        raise RuntimeError("No data downloaded from any chunk.")

    merged = pd.concat(all_px, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    rename_map = {c: str(c).replace(".NS", "") for c in merged.columns}
    merged = merged.rename(columns=rename_map)

    merged = merged.dropna(how="all").ffill().dropna(how="any")
    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "Date"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(merged)}")
    print(f"Symbols retained: {merged.shape[1]}")


if __name__ == "__main__":
    main()
