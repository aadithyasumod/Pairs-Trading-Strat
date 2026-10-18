from __future__ import annotations

import math


def max_holding_days_from_half_life(half_life: float, cap: int, default: int = 10) -> int:
    """Time stop: hold for at most two half-lives, never longer than `cap` bars.

    After two half-lives an OU spread is expected to have closed 75% of its gap, which is
    roughly the distance from a 2.0 entry to a 0.5 exit. If it has not, the relationship is
    reverting slower than the model believes. `default` applies when no half-life is available.
    """
    if half_life is None or not math.isfinite(half_life) or half_life <= 0:
        return max(1, min(default, cap))
    return max(1, min(int(math.ceil(2.0 * half_life)), cap))
