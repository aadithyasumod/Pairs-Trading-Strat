"""Share-based pair backtester.

Timing contract (no look-ahead):
- the decision at bar t uses the spread model at t, which only uses closes up to t
- a signal decided at t fills at the close of bar t + execution_lag (default: the next bar)
- time-based forced exits (session square-off) are known in advance and fill at bar t itself
- PnL on bar t comes from shares held over (t-1, t], marked at closes

Positions are whole shares of both legs, fixed at entry, so the hedge does not silently
rebalance for free. Costs are charged per leg on the notional traded at every fill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .position_sizing import compute_pair_order_size
from .regime import pair_health
from .signals import FLAT, PairStateMachine, SignalRules, spread_model

TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "direction",
    "entry_z",
    "exit_z",
    "hedge_beta",
    "qty_a",
    "qty_b",
    "entry_price_a",
    "entry_price_b",
    "exit_price_a",
    "exit_price_b",
    "gross_notional",
    "bars_held",
    "gross_pnl",
    "costs",
    "net_pnl",
    "return_on_gross",
    "exit_reason",
]
SIGNAL_COLUMNS = ["timestamp", "target", "reason", "zscore", "beta"]


@dataclass
class PairBacktestResult:
    equity_curve: pd.Series  # cumulative net PnL in currency, starting at 0
    pnl: pd.Series  # net PnL per bar
    positions: pd.Series  # direction held at the end of each bar (+1 / -1 / 0)
    trades: pd.DataFrame  # one row per round trip
    signals: pd.DataFrame  # every entry/exit decision, stamped at the decision bar
    metrics: dict
    model: pd.DataFrame  # spread model per bar (alpha, beta, spread, zscore, half_life, health)

    @property
    def n_trades(self) -> int:
        return int(len(self.trades))

    @property
    def sharpe(self) -> float:
        return float(self.metrics["sharpe"])

    @property
    def max_drawdown(self) -> float:
        return float(self.metrics["max_drawdown"])


def performance_metrics(
    pnl: pd.Series,
    trades: pd.DataFrame,
    positions: pd.Series,
    capital: float,
    periods_per_year: float,
) -> dict:
    """Return, risk and trade-quality statistics. Returns are PnL over a fixed capital base."""
    if capital <= 0:
        raise ValueError("capital must be positive.")

    ret = pnl / capital
    ann = math.sqrt(periods_per_year)
    vol = float(ret.std(ddof=0)) if len(ret) else 0.0
    ann_return = float(ret.mean() * periods_per_year) if len(ret) else 0.0
    ann_vol = vol * ann
    downside = float(np.sqrt((np.minimum(ret.to_numpy(), 0.0) ** 2).mean())) * ann if len(ret) else 0.0

    equity = capital + pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    max_dd_pct = float((drawdown / peak).min()) if len(drawdown) else 0.0

    n_trades = int(len(trades))
    net = trades["net_pnl"] if n_trades else pd.Series(dtype=float)
    wins = float(net[net > 0].sum())
    losses = float(net[net < 0].sum())
    if n_trades >= 2:
        trade_ret = trades["return_on_gross"]
        sd = float(trade_ret.std(ddof=1))
        t_stat = float(trade_ret.mean() / (sd / math.sqrt(n_trades))) if sd > 0 else 0.0
    else:
        t_stat = float("nan")

    return {
        "bars": int(len(pnl)),
        "total_pnl": float(pnl.sum()),
        "total_return": float(pnl.sum() / capital),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 1e-12 else 0.0,
        "sortino": ann_return / downside if downside > 1e-12 else 0.0,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "calmar": ann_return / abs(max_dd_pct) if max_dd_pct < 0 else float("nan"),
        "n_trades": n_trades,
        "win_rate": float((net > 0).mean()) if n_trades else float("nan"),
        "avg_trade_pnl": float(net.mean()) if n_trades else float("nan"),
        "profit_factor": (wins / -losses) if losses < 0 else (float("inf") if wins > 0 else float("nan")),
        "avg_bars_held": float(trades["bars_held"].mean()) if n_trades else float("nan"),
        "trade_t_stat": t_stat,
        "total_costs": float(trades["costs"].sum()) if n_trades else 0.0,
        "exposure": float((positions != 0).mean()) if len(positions) else 0.0,
    }


def simulate_pair(
    price_a: pd.Series,
    price_b: pd.Series,
    model: pd.DataFrame,
    rules: SignalRules,
    *,
    notional: float,
    cost_bps_per_leg: float,
    max_hold_cap: int,
    default_hold: int = 10,
    execution_lag: int = 1,
    can_enter: pd.Series | None = None,
    regime_ok: pd.Series | None = None,
    force_exit: pd.Series | None = None,
    capital: float | None = None,
    periods_per_year: float = 252.0,
) -> PairBacktestResult:
    """Run the shared state machine over one pair and account for fills, costs and trades.

    `model` needs zscore, beta and half_life columns on the same index as the prices.
    `can_enter` / `regime_ok` are boolean gates per bar; `force_exit` holds a reason string on
    bars where any position must be closed immediately (for example an intraday square-off).
    """
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0.")
    idx = price_a.index
    n = len(idx)
    if not (len(price_b) == len(model) == n):
        raise ValueError("Prices and model must share the same index.")

    pa = price_a.to_numpy(dtype=float).tolist()
    pb = price_b.to_numpy(dtype=float).tolist()
    zs = model["zscore"].to_numpy(dtype=float).tolist()
    betas = model["beta"].to_numpy(dtype=float).tolist()
    hls = model["half_life"].to_numpy(dtype=float).tolist()
    enter_ok = [True] * n if can_enter is None else can_enter.reindex(idx, fill_value=False).astype(bool).tolist()
    regime = [True] * n if regime_ok is None else regime_ok.reindex(idx, fill_value=True).astype(bool).tolist()
    forced = [None] * n
    if force_exit is not None:
        forced = [r if isinstance(r, str) and r else None for r in force_exit.reindex(idx).tolist()]

    machine = PairStateMachine(rules, max_hold_cap=max_hold_cap, default_hold=default_hold)
    cost_rate = cost_bps_per_leg / 10_000.0

    pnl = np.zeros(n)
    positions = np.zeros(n, dtype=int)
    trades: list[dict] = []
    signals: list[tuple] = []
    pending: list[tuple[int, int, str, float, float]] = []  # (fill_bar, target, reason, z, beta)

    held = FLAT
    qty_a = qty_b = 0
    open_trade: dict | None = None

    def fill(t: int, target: int, reason: str, z_dec: float, beta_dec: float) -> None:
        nonlocal held, qty_a, qty_b, open_trade
        if target == FLAT:
            if held == FLAT:
                return
            cost = (abs(qty_a) * pa[t] + abs(qty_b) * pb[t]) * cost_rate
            pnl[t] -= cost
            gross = qty_a * (pa[t] - open_trade["entry_price_a"]) + qty_b * (pb[t] - open_trade["entry_price_b"])
            costs = open_trade["costs"] + cost
            open_trade.update(
                exit_time=idx[t],
                exit_z=z_dec,
                exit_price_a=pa[t],
                exit_price_b=pb[t],
                bars_held=t - open_trade.pop("_entry_bar"),
                gross_pnl=gross,
                costs=costs,
                net_pnl=gross - costs,
                return_on_gross=(gross - costs) / open_trade["gross_notional"],
                exit_reason=reason,
            )
            trades.append(open_trade)
            open_trade = None
            held, qty_a, qty_b = FLAT, 0, 0
            return

        if held != FLAT:
            return
        size = compute_pair_order_size(pa[t], pb[t], beta_dec, notional)
        if size.qty_a == 0 or size.qty_b == 0:
            return  # capital too small for one share of a leg: skip the trade
        cost = size.gross_notional * cost_rate
        pnl[t] -= cost
        held = target
        qty_a = target * size.qty_a
        qty_b = -target * size.qty_b
        open_trade = {
            "entry_time": idx[t],
            "direction": target,
            "entry_z": z_dec,
            "hedge_beta": beta_dec,
            "qty_a": qty_a,
            "qty_b": qty_b,
            "entry_price_a": pa[t],
            "entry_price_b": pb[t],
            "gross_notional": size.gross_notional,
            "costs": cost,
            "_entry_bar": t,
        }

    for t in range(n):
        if t > 0 and held != FLAT:
            pnl[t] += qty_a * (pa[t] - pa[t - 1]) + qty_b * (pb[t] - pb[t - 1])

        while pending and pending[0][0] == t:
            _, target, reason, z_dec, beta_dec = pending.pop(0)
            fill(t, target, reason, z_dec, beta_dec)

        force_reason = forced[t]
        target, reason = machine.on_bar(
            zs[t], hls[t], betas[t], can_enter=enter_ok[t], regime_ok=regime[t], force_reason=force_reason
        )
        if reason not in ("hold", "flat"):
            signals.append((idx[t], target, reason, zs[t], betas[t]))

        if force_reason is not None:
            if held != FLAT or pending:
                pending.clear()
                fill(t, FLAT, force_reason, zs[t], betas[t])
        elif reason == "entry" or reason.startswith("exit"):
            if execution_lag == 0:
                fill(t, target, reason, zs[t], betas[t])
            elif t + execution_lag < n:
                pending.append((t + execution_lag, target, reason, zs[t], betas[t]))

        positions[t] = held

    if held != FLAT:
        fill(n - 1, FLAT, "end_of_data", zs[-1], betas[-1])
        positions[-1] = FLAT

    pnl_s = pd.Series(pnl, index=idx, name="pnl")
    pos_s = pd.Series(positions, index=idx, name="position")
    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    signals_df = pd.DataFrame(signals, columns=SIGNAL_COLUMNS)
    metrics = performance_metrics(pnl_s, trades_df, pos_s, capital or notional, periods_per_year)
    return PairBacktestResult(
        equity_curve=pnl_s.cumsum().rename("equity"),
        pnl=pnl_s,
        positions=pos_s,
        trades=trades_df,
        signals=signals_df,
        metrics=metrics,
        model=model,
    )


def daily_rules(cfg: StrategyConfig) -> SignalRules:
    return SignalRules(cfg.z_entry, cfg.z_exit, cfg.z_stop, cfg.cooldown_bars_after_stop)


def build_daily_model(prices_a: pd.Series, prices_b: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    model = spread_model(prices_a, prices_b, cfg.hedge_window, cfg.zscore_window, cfg.ou_window)
    model["health"] = pair_health(
        prices_a,
        prices_b,
        model,
        window=cfg.hedge_window,
        min_corr=cfg.min_rolling_corr,
        max_beta_jump=cfg.max_beta_jump,
        max_vol_spike_ratio=cfg.max_vol_spike_ratio,
    )
    return model


def backtest_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    cfg: StrategyConfig | None = None,
    *,
    trade_start: pd.Timestamp | None = None,
    trade_end: pd.Timestamp | None = None,
) -> PairBacktestResult:
    """Daily pair backtest: spread model -> health gate -> shared rules -> share-based fills.

    Bars before `trade_start` only warm up the rolling models (no trading, not in metrics).
    Any position still open at `trade_end` (or the last bar) is closed there.
    """
    cfg = cfg or StrategyConfig()
    px = pd.concat([prices_a.rename("a"), prices_b.rename("b")], axis=1).dropna()
    if trade_end is not None:
        px = px[px.index <= pd.Timestamp(trade_end)]
    min_bars = cfg.hedge_window + cfg.zscore_window + 2
    if len(px) < min_bars:
        raise ValueError(f"Need at least {min_bars} overlapping bars for this pair, got {len(px)}.")

    model = build_daily_model(px["a"], px["b"], cfg)
    can_enter = model["health"] >= cfg.health_pause_threshold
    if trade_start is not None:
        can_enter &= px.index >= pd.Timestamp(trade_start)

    res = simulate_pair(
        px["a"],
        px["b"],
        model,
        daily_rules(cfg),
        notional=cfg.notional_per_trade,
        cost_bps_per_leg=cfg.fee_bps + cfg.slippage_bps,
        max_hold_cap=cfg.max_holding_days_cap,
        default_hold=10,
        execution_lag=cfg.execution_lag_bars,
        can_enter=can_enter,
        capital=cfg.notional_per_trade,
        periods_per_year=252.0,
    )
    if trade_start is None:
        return res

    keep = res.pnl.index >= pd.Timestamp(trade_start)
    pnl = res.pnl[keep]
    positions = res.positions[keep]
    return PairBacktestResult(
        equity_curve=pnl.cumsum().rename("equity"),
        pnl=pnl,
        positions=positions,
        trades=res.trades,
        signals=res.signals,
        metrics=performance_metrics(pnl, res.trades, positions, cfg.notional_per_trade, 252.0),
        model=res.model[keep],
    )
