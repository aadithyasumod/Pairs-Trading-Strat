# Pairs Trading Lab

[![CI](https://github.com/aadithyasumod/Pairs-Trading-Strat/actions/workflows/ci.yml/badge.svg)](https://github.com/aadithyasumod/Pairs-Trading-Strat/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white) ![statsmodels](https://img.shields.io/badge/statsmodels-cointegration-4B8BBE) ![Zerodha](https://img.shields.io/badge/Zerodha-Kite%20Connect-387ED1) ![License](https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey)

A research framework for NSE equity pairs trading: it finds cointegrated pairs across the full NSE universe, trades their spread with one rule book shared by the backtester, the walk-forward test and a paper/live trader, and runs a monitor that flags today's deployable pairs.

```
$ make run-walkforward
Data: synthetic universe (P* pairs are cointegrated, D* are decoys, S* are singles) | 14 symbols
Window 1: trade 2022-01-05 -> 2022-06-29 | 2 pairs | 7 trades | PnL 73,410
...
Out-of-sample portfolio (what you would have earned)
  sharpe                   2.0844
  max_drawdown_pct         -0.0117
  n_trades                 86
  win_rate                 0.7209
  walk_forward_efficiency  1.1034
```
*(synthetic data with planted pairs, used to check the method finds them; not a real-market result)*

## What it does

- **Selects pairs** whose log prices are cointegrated (Engle-Granger p ≤ 0.05, Johansen check), with a positive hedge ratio and a half-life between 1 and 60 bars. Correlation is only a speed prefilter.
- **Models the spread** causally: rolling OLS `log(A) = α + β·log(B)` fitted on bars up to *t*−1, then a rolling z-score.
- **Trades one rule book** (`PairStateMachine`): enter when `z_entry ≤ |z| < z_stop`, exit on mean reversion, stop when the spread keeps widening or after 2 × half-life bars, then cool down. A regime-health score (rolling correlation, beta jumps, volatility spikes) gates new entries.
- **Sizes legs** in whole shares with beta-weighted notional, fixed for the trade's life.
- **Backtests honestly**: signals on bar *t* fill at bar *t*+1, costs per leg per side, and a **walk-forward** mode that selects pairs on a 252-day formation window and trades the next 126 days, so all reported PnL is out-of-sample. A bar-by-bar replay through the live trader reproduces the backtest's decisions (tested).
- **Scales to the whole NSE universe** (2,500+ symbols, ~3.1M possible pairs) by filtering bad symbols, building correlation neighbours and running cointegration only on candidates.
- **Monitors live**: health score, GREEN/YELLOW/RED status, data-age guard, top opportunities with A/B/C quality tiers, optional webhook alert and an Excel workbook per run.
- **Intraday mode**: entries 09:30–15:00, everything squared off at 15:20.
- **Zerodha adapter** in paper mode by default; live mode reads `KITE_API_KEY` and `KITE_ACCESS_TOKEN` from the environment.

## Architecture

```
fetch_nse_symbols / build_universe_from_zerodha  ->  symbol list
download_daily / download_nse_universe           ->  wide daily price CSV (yfinance)
pair_selection    correlation prefilter -> Engle-Granger + Johansen -> half-life screen
hedge_ratio + signals (PairStateMachine) + regime + position_sizing
  ├─ backtest / walk_forward / batch_daily / universe_scan   (daily research)
  ├─ intraday_engine / run_intraday / batch_intraday          (intraday research)
  ├─ simple_monitor  ->  console, webhook, results/run_NNNN.xlsx
  └─ live_trader  ->  zerodha_adapter (paper by default)
```

## Requirements

- Python 3.11+ (developed on 3.14)
- Network access to Yahoo Finance and nseindia.com (Kite Connect credentials only for the Zerodha universe builder or live mode)

## Run locally

```bash
make setup                 # create .venv and install requirements
make test                  # full suite: models, rules, accounting, look-ahead, live/backtest parity
make run-walkforward       # out-of-sample walk-forward on a synthetic universe with known answers
```

Real data:

```bash
.venv/bin/python -m pairs_trading.fetch_nse_symbols --out nse_symbols_full.csv
.venv/bin/python -m pairs_trading.download_nse_universe --symbols-file nse_symbols_full.csv \
    --symbol-column Symbol --start 2020-01-01 --out daily_prices_nse_universe.csv
.venv/bin/python -m pairs_trading.universe_scan --csv daily_prices_nse_universe.csv --top 50 \
    --out nse_universe_pair_ranking.csv
```

Point `"csv"` in `monitor_config.json` at your downloaded price file, then:

| Command | What it does |
|---|---|
| `make run-analysis` | one monitor cycle (set `loop_minutes` to repeat) |
| `make run-baseline` | baseline run on synthetic daily data |
| `.venv/bin/python -m pairs_trading.walk_forward --csv daily_prices.csv --top-pairs 5` | walk-forward on real prices |
| `.venv/bin/python -m pairs_trading.run_intraday --csv intraday.csv --asset-a HDFCBANK --asset-b ICICIBANK --plot` | intraday backtest on a `timestamp,SYM1,SYM2` CSV |
| `.venv/bin/python scripts/export_walk_forward.py --csv daily_prices.csv` | walk-forward results as an Excel workbook |

### Optional environment variables

| Variable | Used for |
|---|---|
| `KITE_API_KEY`, `KITE_ACCESS_TOKEN` | Zerodha symbol master and live mode |
| `PAIRS_ALERT_WEBHOOK_URL` | monitor alerts |

## Project structure

```
pairs_trading/       selection, hedge ratio, signals, regime, sizing, backtests, monitor, live trader
scripts/             export_walk_forward.py
tests/               unittest suite (also runs under pytest)
monitor_config.json  monitor thresholds and data source
daily_prices.csv     small sample of bank-stock daily closes
nse_symbols*.csv     NSE symbol lists (fallback and full)
Makefile, requirements.txt
```

Downloaded universe prices, rankings, monitor output and `results/` are generated locally and not tracked.

## Caveats

- Research and execution skeleton, not production capital-deployment code; use paper mode first. Not investment advice.
- India allows cash-equity shorts only intraday (MIS), so a multi-day pair needs stock futures or SLB for the short leg; lot sizes are not modelled.
- Universe CSVs are built from today's symbol list, so delisted names are missing (survivorship bias) and forward-filled gaps can look like flat prices.

## License

Copyright © 2026 aadithyasumod. All rights reserved.

No license is granted. The code is published for reference: you may view it, but not use, copy, modify or distribute it without written permission.
