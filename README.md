# Intraday NIFTY Pairs Trading (Beginner-Friendly, Zerodha-Oriented)

This repo is a full starter framework for the strategy you asked for:

- Intraday pairs trading for equities
- Cointegration-based pair discovery
- Dynamic hedge ratio (rolling OLS)
- OU half-life estimate
- Regime-break detection (health score)
- Risk controls + cooldown + stop
- Backtest engine and a live-trader skeleton
- Zerodha adapter (paper mode by default)

## Important Scope Notes

1. This is **research + execution skeleton**, not production-ready capital deployment code.
2. For live trading, use **paper mode first**, then small size, then scale.
3. For equities shorting intraday, ensure your broker-side product/margin/shorting constraints are satisfied each day.

## Project Structure

- `pairs_trading/config.py`: Daily-mode defaults (kept for simple baseline).
- `pairs_trading/intraday_config.py`: Intraday-specific parameters.
- `pairs_trading/pair_selection.py`: Correlation + Engle-Granger + Johansen filters.
- `pairs_trading/hedge_ratio.py`: Rolling OLS alpha/beta estimator.
- `pairs_trading/signals.py`: Spread, z-score, entry/exit/stop rules.
- `pairs_trading/ou.py`: Half-life estimation from spread dynamics.
- `pairs_trading/regime.py`: Real-time relationship stability score.
- `pairs_trading/position_sizing.py`: Pair quantity sizing.
- `pairs_trading/intraday_engine.py`: Intraday backtest loop.
- `pairs_trading/run_intraday.py`: Intraday backtest CLI.
- `pairs_trading/zerodha_adapter.py`: Zerodha integration adapter.
- `pairs_trading/live_trader.py`: Stateful live-trading logic scaffold.

## Setup

```bash
cd /Users/aadithyasumod/Documents/Playground
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## A) Baseline Run (Synthetic Daily Data)

```bash
python -m pairs_trading.main
```

Useful for quickly understanding the logic chain before intraday.

## B) Intraday Backtest Run (Your Real Data)

Prepare one CSV with this schema:

```csv
timestamp,HDFCBANK,ICICIBANK
2026-01-02 09:15:00,1721.5,1298.2
2026-01-02 09:20:00,1722.1,1298.0
2026-01-02 09:25:00,1721.9,1298.7
```

Run:

```bash
python -m pairs_trading.run_intraday \
  --csv /absolute/path/intraday_pair.csv \
  --asset-a HDFCBANK \
  --asset-b ICICIBANK \
  --plot
```

This prints:
- Total trades
- Final PnL
- Approx annualized intraday Sharpe
- Max drawdown
- Average regime health

## C) Pair Discovery on Multi-Symbol Data

If you have wide daily data (`Date, HDFCBANK, ICICIBANK, ...`), run:

```bash
python -m pairs_trading.main --csv /absolute/path/daily_prices.csv
```

It auto-picks the strongest pair under current filters.

If you do not have a daily CSV yet, download one directly:

```bash
python -m pairs_trading.download_daily \
  --symbols HDFCBANK ICICIBANK AXISBANK SBIN KOTAKBANK \
  --start 2020-01-01 \
  --out /Users/aadithyasumod/Documents/Playground/daily_prices.csv
```

Then run:

```bash
python -m pairs_trading.batch_daily \
  --csv /Users/aadithyasumod/Documents/Playground/daily_prices.csv \
  --top 20 \
  --out /Users/aadithyasumod/Documents/Playground/daily_pair_ranking.csv
```

## C2) Batch Ranking: Test Many Pairs Automatically

### Daily wide CSV

Evaluate prefiltered candidates:

```bash
python -m pairs_trading.batch_daily \
  --csv /absolute/path/daily_prices.csv \
  --top 20 \
  --out /absolute/path/daily_pair_ranking.csv
```

Evaluate every possible pair (slower):

```bash
python -m pairs_trading.batch_daily \
  --csv /absolute/path/daily_prices.csv \
  --all-combinations \
  --top 30 \
  --out /absolute/path/daily_all_pairs.csv
```

### Intraday wide CSV

CSV format must be:
`timestamp,SYM1,SYM2,SYM3,...`

Run:

```bash
python -m pairs_trading.batch_intraday \
  --csv /absolute/path/intraday_prices.csv \
  --min-bars 500 \
  --top 20 \
  --out /absolute/path/intraday_pair_ranking.csv
```

This prints top pairs and writes a full ranking with:
- Sharpe
- Final PnL
- Max drawdown
- Trades
- Average health score

## C3) Full NSE Universe (2500+ Symbols) Workflow

Do not brute-force all pair combinations directly.
For 2500 symbols, total pairs are about 3.1 million.

Use the scalable pipeline below.

### Step 1: Get NSE symbol master file

Create a symbols file (`CSV` or `TXT`) with one symbol per row.
Example CSV:

```csv
Symbol
HDFCBANK
ICICIBANK
SBIN
...
```

### Step 2: Download daily history in chunks

```bash
python -m pairs_trading.download_nse_universe \
  --symbols-file /absolute/path/nse_symbols.csv \
  --symbol-column Symbol \
  --start 2020-01-01 \
  --chunk-size 100 \
  --sleep-seconds 0.6 \
  --out /Users/aadithyasumod/Documents/Playground/daily_prices_nse_universe.csv
```

### Step 3: Run scalable universe scan

```bash
python -m pairs_trading.universe_scan \
  --csv /Users/aadithyasumod/Documents/Playground/daily_prices_nse_universe.csv \
  --min-history 252 \
  --min-corr 0.65 \
  --top-neighbors 20 \
  --max-candidates 40000 \
  --max-backtests 2000 \
  --top 50 \
  --out /Users/aadithyasumod/Documents/Playground/nse_universe_pair_ranking.csv
```

This flow:
1. Filters bad/incomplete symbols.
2. Builds correlation-neighbor candidates.
3. Runs cointegration only on filtered pairs.
4. Backtests only best cointegrated candidates.

This is the practical way to handle full-universe scanning.

## D) Zerodha Integration (Paper/Live)

### 1) Paper mode first

`ZerodhaAdapter(paper=True)` is default in `live_trader.py`.
It prints intended orders without sending them.

### 2) Live mode env vars

```bash
export KITE_API_KEY="your_key"
export KITE_ACCESS_TOKEN="your_access_token"
```

Then instantiate adapter as `ZerodhaAdapter(paper=False)`.

## How the Intraday Model Works

1. Build rolling hedge ratio: `log(A) = alpha + beta * log(B)`.
2. Spread = `log(A) - alpha - beta*log(B)`.
3. Compute rolling z-score of spread.
4. Enter when spread is extreme (`|z| >= z_entry`).
5. Exit on mean reversion (`|z| <= z_exit`).
6. Stop on extreme (`|z| >= z_stop`) or regime deterioration.
7. Regime health uses:
- rolling return correlation
- beta jump instability
- spread volatility spike
8. Size both legs with beta-adjusted notional.

## Beginner Operating Plan (Recommended)

1. Run intraday backtest for one pair first.
2. Tune only 3 parameters initially:
- `z_entry`
- `z_exit`
- `hedge_window_bars`
3. Validate after costs.
4. Add 2–3 more pairs only after one pair is stable.
5. Run paper mode for at least 2 weeks.
6. Go live at minimum size.

## Practical Production Upgrades (Next)

1. Replace rolling OLS with Kalman filter beta.
2. Add true Johansen rolling test in live gate.
3. Replace placeholder flatten logic with actual position-sync from broker.
4. Add exchange calendar, no-trade windows, and order retry logic.
5. Add comprehensive trade journal and PnL attribution.

## Risk Notes

- Intraday correlation can break quickly around macro events.
- Equity short availability and costs can vary intraday.
- Backtest slippage is usually optimistic.
- Never rely on a single pair.
