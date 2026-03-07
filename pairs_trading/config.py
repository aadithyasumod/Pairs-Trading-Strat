from dataclasses import dataclass


@dataclass
class StrategyConfig:
    # Pair selection
    min_history: int = 252
    correlation_window: int = 120
    min_correlation: float = 0.6
    max_pairs_to_check: int = 20

    # Modeling
    hedge_window: int = 120
    zscore_window: int = 60
    ou_window: int = 120

    # Signals
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5

    # Risk
    max_holding_days_cap: int = 20
    notional_per_trade: float = 1_000_000.0

    # Simple cost model
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
