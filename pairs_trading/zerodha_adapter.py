from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderIntent:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    product: str = "MIS"
    order_type: str = "MARKET"
    exchange: str = "NSE"


class ZerodhaAdapter:
    """
    Thin adapter around KiteConnect.

    - In paper mode, it only logs orders.
    - In live mode, it requires kiteconnect and valid session credentials.
    """

    def __init__(self, paper: bool = True) -> None:
        self.paper = paper
        self.kite = None

        if not paper:
            try:
                from kiteconnect import KiteConnect
            except Exception as exc:  # pragma: no cover - depends on local install
                raise RuntimeError("kiteconnect not installed. Run: pip install -r requirements.txt") from exc

            api_key = os.getenv("KITE_API_KEY")
            access_token = os.getenv("KITE_ACCESS_TOKEN")
            if not api_key or not access_token:
                raise RuntimeError("Set KITE_API_KEY and KITE_ACCESS_TOKEN env vars for live mode.")

            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)

    def place_order(self, order: OrderIntent) -> str:
        if order.side not in ("BUY", "SELL"):
            raise ValueError(f"Order side must be BUY or SELL, got {order.side!r}.")
        if order.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {order.quantity}.")
        if self.paper:
            print(
                f"[PAPER ORDER] {order.side} {order.quantity} {order.symbol} "
                f"{order.exchange} {order.product} {order.order_type}"
            )
            return "paper-order-id"

        txn = self.kite.TRANSACTION_TYPE_BUY if order.side == "BUY" else self.kite.TRANSACTION_TYPE_SELL
        order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=order.exchange,
            tradingsymbol=order.symbol,
            transaction_type=txn,
            quantity=order.quantity,
            product=order.product,
            order_type=order.order_type,
        )
        return str(order_id)
