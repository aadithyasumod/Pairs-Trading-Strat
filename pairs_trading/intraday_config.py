from dataclasses import dataclass


@dataclass
class IntradayConfig:
    # Market and bars
    bar_minutes: int = 5
    market_open: str = "09:15"
    market_close: str = "15:30"
    square_off_time: str = "15:20"  # MIS positions are flattened at or after this bar
    no_new_entries_before: str = "09:30"  # skip the opening auction noise
    no_new_entries_after: str = "15:00"  # leave room to exit before square-off

    # Pair modeling windows in bars
    hedge_window_bars: int = 120
    zscore_window_bars: int = 60
    ou_window_bars: int = 120

    # Signal thresholds
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5

    # Regime health thresholds
    min_rolling_corr: float = 0.45
    max_beta_jump: float = 0.08
    max_vol_spike_ratio: float = 2.0
    health_pause_threshold: float = 0.4  # no new entries below this
    health_exit_threshold: float = 0.25  # force an exit only below this (hysteresis avoids whipsaw)

    # Risk and sizing
    capital: float = 1_000_000.0
    max_capital_per_pair: float = 200_000.0
    max_gross_leverage: float = 1.5
    max_holding_bars_cap: int = 120

    # Execution. Signals use the close of bar t and fill at the close of bar t + lag.
    execution_lag_bars: int = 1

    # Costs per leg, per side, in bps of traded notional (rough MIS placeholders)
    fee_bps_per_leg: float = 5.0
    slippage_bps_per_leg: float = 4.0

    # Safeguards
    cooldown_bars_after_stop: int = 12
