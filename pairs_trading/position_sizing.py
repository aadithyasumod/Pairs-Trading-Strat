from __future__ import annotations

import math
from dataclasses import dataclass

# Hedge ratios outside this band make one leg dominate the pair, turning it into a directional bet.
MIN_HEDGE_BETA = 0.1
MAX_HEDGE_BETA = 5.0


@dataclass
class PairOrderSize:
    qty_a: int
    qty_b: int
    notional_a: float
    notional_b: float
    gross_notional: float


def hedge_ratio_ok(beta: float) -> bool:
    """True when beta is a usable hedge ratio for a long/short pair."""
    return math.isfinite(beta) and MIN_HEDGE_BETA <= beta <= MAX_HEDGE_BETA


def compute_pair_order_size(
    price_a: float,
    price_b: float,
    beta: float,
    pair_capital: float,
) -> PairOrderSize:
    """
    Beta-weighted leg sizing.

    The spread is log(A) - beta * log(B), so one unit of spread exposure is
    notional_a = pair_capital of A against notional_b = beta * pair_capital of B.
    Quantities round down to whole shares; a zero quantity means the capital cannot buy
    one share of that leg and the trade should be skipped.
    """
    if not hedge_ratio_ok(beta):
        raise ValueError(
            f"Hedge ratio {beta!r} is outside [{MIN_HEDGE_BETA}, {MAX_HEDGE_BETA}]; refusing to size a mis-hedged pair."
        )
    if price_a <= 0 or price_b <= 0:
        raise ValueError("Prices must be positive to size orders.")
    if pair_capital <= 0:
        raise ValueError("pair_capital must be positive.")

    qty_a = int(pair_capital / price_a)
    qty_b = int(pair_capital * beta / price_b)

    actual_notional_a = qty_a * price_a
    actual_notional_b = qty_b * price_b

    return PairOrderSize(
        qty_a=qty_a,
        qty_b=qty_b,
        notional_a=actual_notional_a,
        notional_b=actual_notional_b,
        gross_notional=actual_notional_a + actual_notional_b,
    )
