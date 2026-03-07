from __future__ import annotations

import argparse

from .data import load_intraday_pair_csv
from .intraday_config import IntradayConfig
from .live_trader import IntradayPairsTrader
from .zerodha_adapter import ZerodhaAdapter



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay intraday bars through live-trader logic")
    p.add_argument("--csv", required=True)
    p.add_argument("--asset-a", required=True)
    p.add_argument("--asset-b", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = IntradayConfig()

    px = load_intraday_pair_csv(args.csv, args.asset_a, args.asset_b)
    trader = IntradayPairsTrader(
        asset_a=args.asset_a,
        asset_b=args.asset_b,
        cfg=cfg,
        adapter=ZerodhaAdapter(paper=True),
    )

    # Feed bars one by one to mimic live updates.
    for i in range(len(px)):
        trader.on_new_history(px.iloc[: i + 1])

    print("Replay finished. Check [PAPER ORDER] lines above for generated actions.")


if __name__ == "__main__":
    main()
