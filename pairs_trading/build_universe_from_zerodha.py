from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full NSE equity universe from Zerodha instruments")
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "nse_symbols_full.csv"),
        help="Output symbols CSV path",
    )
    p.add_argument(
        "--segment",
        default="NSE",
        choices=["NSE", "BSE"],
        help="Exchange segment to pull instruments from",
    )
    return p.parse_args()



def main() -> None:
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key or not access_token:
        raise SystemExit(
            "Missing KITE_API_KEY or KITE_ACCESS_TOKEN. "
            "Set both env vars, then rerun this command."
        )

    try:
        from kiteconnect import KiteConnect
    except Exception as exc:
        raise SystemExit("kiteconnect not installed. Run: pip install -r requirements.txt") from exc

    args = parse_args()

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    ins = pd.DataFrame(kite.instruments(args.segment))
    if ins.empty:
        raise SystemExit("No instruments returned from Zerodha API.")

    # Cash equities only
    eq = ins[(ins["segment"] == args.segment) & (ins["instrument_type"] == "EQ")].copy()
    if eq.empty:
        raise SystemExit("No EQ instruments found in response.")

    symbols = sorted(eq["tradingsymbol"].dropna().astype(str).str.upper().unique().tolist())
    out = pd.DataFrame({"Symbol": symbols})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Total symbols: {len(out)}")
    print(f"Segment: {args.segment}")


if __name__ == "__main__":
    main()
