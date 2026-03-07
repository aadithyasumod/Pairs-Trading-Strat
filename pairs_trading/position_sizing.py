from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PairOrderSize:
    qty_a: int
    qty_b: int
    notional_a: float
    notional_b: float
    gross_notional: float



def compute_pair_order_size(
    price_a: float,
    price_b: float,
    beta: float,
    pair_capital: float,
) -> PairOrderSize:
    """
    Dollar-neutral-ish leg sizing using dynamic beta.

    notional_a = pair_capital
    notional_b = abs(beta) * pair_capital
    """
    beta_abs = abs(beta) if math.isfinite(beta) else 1.0
    beta_abs = max(0.1, min(beta_abs, 5.0))

    notional_a = pair_capital
    notional_b = pair_capital * beta_abs

    qty_a = max(1, int(notional_a / max(price_a, 1e-9)))
    qty_b = max(1, int(notional_b / max(price_b, 1e-9)))

    actual_notional_a = qty_a * price_a
    actual_notional_b = qty_b * price_b

    return PairOrderSize(
        qty_a=qty_a,
        qty_b=qty_b,
        notional_a=actual_notional_a,
        notional_b=actual_notional_b,
        gross_notional=actual_notional_a + actual_notional_b,
    )
