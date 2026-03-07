from __future__ import annotations

import math



def max_holding_days_from_half_life(half_life: float, cap: int) -> int:
    if half_life is None or not math.isfinite(half_life) or half_life <= 0:
        return min(10, cap)
    return min(int(math.ceil(2.0 * half_life)), cap)
