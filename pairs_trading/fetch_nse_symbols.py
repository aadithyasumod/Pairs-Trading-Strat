from __future__ import annotations

import argparse
import io
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URLS = [
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch full NSE equity symbol list")
    p.add_argument("--url", default=None, help="Single NSE symbol master CSV URL override")
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "nse_symbols_full.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--fallback-local",
        default=str(PROJECT_ROOT / "nse_symbols.csv"),
        help="Optional local symbol CSV used if network fetch fails",
    )
    p.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable SSL certificate verification (last resort)",
    )
    return p.parse_args()



def _extract_symbols(df: pd.DataFrame) -> pd.Series:
    symbol_col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
    syms = (
        df[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
    )
    return syms



def _download_csv_as_df(url: str, insecure_ssl: bool) -> pd.DataFrame:
    if insecure_ssl:
        ctx = ssl._create_unverified_context()
    else:
        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()

    with urllib.request.urlopen(url, context=ctx, timeout=20) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw))


def _try_remote(urls: list[str], insecure_ssl: bool) -> pd.Series:
    last_err = None
    for url in urls:
        try:
            df = _download_csv_as_df(url, insecure_ssl=insecure_ssl)
            syms = _extract_symbols(df)
            if len(syms) > 0:
                print(f"Fetched symbols from: {url}")
                return syms
        except Exception as exc:
            last_err = exc
            print(f"Fetch failed for {url}: {type(exc).__name__}")
    if last_err:
        raise last_err
    raise RuntimeError("No symbol source URLs were provided.")



def _try_local(path: str) -> pd.Series | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "Symbol" in df.columns:
            syms = df["Symbol"]
        else:
            syms = df.iloc[:, 0]
        syms = (
            syms.dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
        )
        if len(syms) > 0:
            print(f"Using local fallback symbols from: {p}")
            return syms
    except Exception:
        return None
    return None



def main() -> None:
    args = parse_args()
    urls = [args.url] if args.url else DEFAULT_URLS

    syms = None
    try:
        syms = _try_remote(urls, insecure_ssl=args.insecure_ssl)
    except Exception as exc:
        print(f"Remote fetch unavailable: {type(exc).__name__}: {exc}")
        syms = _try_local(args.fallback_local)
        if syms is None:
            raise SystemExit(
                "Could not fetch NSE symbols from network and no valid local fallback found. "
                "Provide internet access or create a local symbols CSV and pass --fallback-local."
            )

    out = pd.DataFrame({"Symbol": syms})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Total symbols: {len(out)}")


if __name__ == "__main__":
    main()
