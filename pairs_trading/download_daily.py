from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SYMBOLS = [
    "HDFCBANK",
    "ICICIBANK",
    "AXISBANK",
    "SBIN",
    "KOTAKBANK",
    "INDUSINDBK",
    "BAJFINANCE",
    "BAJAJFINSV",
]



def _to_yahoo_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.endswith(".NS"):
        return s
    return f"{s}.NS"



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily NSE prices and save wide CSV")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="NSE symbols without suffix, e.g. HDFCBANK ICICIBANK",
    )
    p.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "daily_prices.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--column",
        choices=["Close", "Adj Close"],
        default="Close",
        help="Price column to export",
    )
    p.add_argument(
        "--no-auto-adjust",
        action="store_true",
        help="Disable auto-adjust (default uses adjusted prices through auto_adjust=True)",
    )
    return p.parse_args()



def main() -> None:
    args = parse_args()

    yahoo_symbols = [_to_yahoo_symbol(s) for s in args.symbols]
    auto_adjust = not args.no_auto_adjust

    data = yf.download(
        yahoo_symbols,
        start=args.start,
        end=args.end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if data.empty:
        raise RuntimeError("No data downloaded. Check symbols/date range/internet connectivity.")

    if auto_adjust:
        if "Close" not in data.columns.get_level_values(0):
            raise RuntimeError("Downloaded data missing 'Close' field.")
        px = data["Close"].copy()
    else:
        if args.column not in data.columns.get_level_values(0):
            raise RuntimeError(f"Downloaded data missing '{args.column}' field.")
        px = data[args.column].copy()

    # Map back to plain NSE symbols without .NS suffix
    rename_map = {ys: ys.replace(".NS", "") for ys in px.columns}
    px = px.rename(columns=rename_map)

    px = px.dropna(how="all").ffill().dropna(how="any")
    px.index = pd.to_datetime(px.index)
    px.index.name = "Date"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    px.reset_index().to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(px)}")
    print(f"Symbols: {list(px.columns)}")


if __name__ == "__main__":
    main()
