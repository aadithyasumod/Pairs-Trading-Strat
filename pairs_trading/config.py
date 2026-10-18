from dataclasses import dataclass


@dataclass
class StrategyConfig:
    # Pair selection (formation period)
    min_history: int = 252
    correlation_window: int = 120
    min_correlation: float = 0.6
    max_pairs_to_check: int = 20
    coint_p_threshold: float = 0.05
    min_half_life: float = 1.0
    max_half_life: float = 60.0

    # Modeling (rolling windows, in bars)
    hedge_window: int = 120
    zscore_window: int = 60
    ou_window: int = 120

    # Signals
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5

    # Risk
    max_holding_days_cap: int = 20
    cooldown_bars_after_stop: int = 5
    notional_per_trade: float = 1_000_000.0

    # Regime health gate: no new entries while the relationship looks unstable
    min_rolling_corr: float = 0.45
    max_beta_jump: float = 0.08
    max_vol_spike_ratio: float = 2.0
    health_pause_threshold: float = 0.4

    # Execution. Signals use the close of bar t and fill at the close of bar t + lag.
    # Costs are per leg, per side, in bps of traded notional (NSE stock futures ballpark).
    execution_lag_bars: int = 1
    fee_bps: float = 5.0
    slippage_bps: float = 3.0

    # Walk-forward: select pairs on `formation_days`, trade them for the next `trading_days`
    formation_days: int = 252
    trading_days: int = 126
    walk_forward_top_pairs: int = 5
