# Walk-Forward Trading App — Build Brief v2

---

## Stack

| Library | Role |
|---|---|
| `streamlit` | UI framework |
| `plotly` | All charts and visualizations |
| `yfinance` | Market data fetching |
| `scipy` | Butterworth filter |
| `pandas` / `numpy` | Data computation |
| `pandas-ta` | Indicator calculations inside strategies |

No embedded AI. No database. No authentication.

---

## Project Structure

```
trading_app/
├── app.py                  # Main Streamlit entry point
├── backtester.py           # Walk-forward engine
├── filters.py              # Butterworth filter (lfilter, causal only)
├── strategies/
│   ├── __init__.py         # Auto-discovers strategy files
│   ├── rsi_sma.py          # Default bundled strategy
│   └── ...                 # User-imported files drop here
├── components/
│   ├── sidebar.py          # All parameter controls
│   ├── charts.py           # All Plotly chart builders
│   └── strategy_card.py    # Plain-English strategy info panel
└── requirements.txt
```

---

## Strategy Contract

Every strategy file must export exactly this interface. The app auto-discovers files in the `strategies/` folder by reading `STRATEGY_INFO` and registering `generate_signals`.

```python
STRATEGY_INFO = {
    "name": "RSI + SMA Filter",
    "description": (
        "Goes long when RSI crosses above 30 and price is above SMA(20). "
        "Exits when RSI crosses above 70."
    ),
    "parameters": {
        "rsi_period": 14,
        "sma_period": 20,
        "oversold": 30,
        "overbought": 70
    },
    "parameter_ranges": {
        "rsi_period": {"min": 5,  "max": 30,  "step": 1},
        "sma_period": {"min": 10, "max": 50,  "step": 1},
        "oversold":   {"min": 20, "max": 40,  "step": 1},
        "overbought": {"min": 60, "max": 80,  "step": 1}
    },
    "warmup_bars": 50,   # bars to discard at fold start for indicator warm-up ONLY
                         # the engine adds its own filter warm-up on top of this value
    "tags": ["mean-reversion", "long-only", "trend-filter"]
}

def generate_signals(df: pd.DataFrame, **params) -> pd.Series:
    """
    df guaranteed columns: Open, High, Low, Close, Volume (daily OHLCV)

    IMPORTANT — column treatment by the engine:
      df["Close"]  : Butterworth-filtered close price (causal, lfilter).
                     Use this for all indicator calculations.
      df["Open"]   : raw open price (unfiltered)
      df["High"]   : raw high price (unfiltered)
      df["Low"]    : raw low price (unfiltered)
      df["Volume"] : raw volume (unfiltered)
    Do NOT mix filtered Close with raw High/Low for range-based logic
    (e.g. ATR, true range) without acknowledging the inconsistency.
    The warm-up rows have already been discarded before this function
    is called — the first row of df is always safe to use.

    Returns: pd.Series of int, same index as df
      1  = buy / hold long
     -1  = sell / short
      0  = flat
    RULE: No lookahead — only use data available at row i to decide row i.
    """
    pass
```

### Auto-discovery (`strategies/__init__.py`)

Scans the strategies folder, imports every `.py` file, reads `STRATEGY_INFO`, and registers each strategy. A user dropping a new file into the folder will see it appear in the dropdown on the next run.

### Validation on import

Before accepting a user-uploaded `.py` file, the app checks that:
- `STRATEGY_INFO` dict exists and contains `name`, `description`, `parameters`, `tags`
- `STRATEGY_INFO` contains `parameter_ranges` with a `min`, `max`, and `step` entry for every key in `parameters`
- `STRATEGY_INFO` contains `warmup_bars` as a positive integer
- `generate_signals` callable exists with the correct signature

If any field is absent or incomplete, the app rejects the file with a clear error message explaining what is missing. There is no silent fallback.

> ⚠️ **Security note**: Importing a `.py` file executes arbitrary code. This app is intended for local use only. Never deploy it as a shared or public service without adding proper sandboxing (e.g. `subprocess` isolation or `RestrictedPython`).

---

## UI Layout

Nothing runs until the **Run Backtest** button is clicked. The backtest never triggers on slider or input change.

### Sidebar — 5 Collapsible Sections

#### 1. Asset & Data
- Ticker text input (default: `SPY`)
- Start date / End date pickers (default: `2018-01-01` → today)
- Price field selector: Close / Adjusted Close

#### 2. Capital & Position Sizing
- **Initial capital**: dollar input (default: `$10,000`)
  - Starting portfolio value; all equity curves and P&L figures are denominated in dollars
- **Position sizing**: radio selector with two modes:
  - `% of equity` (default: `100%`) — each trade allocates a fixed percentage of current equity; compound growth, position size changes over time
  - `Fixed $ per trade` (default: `$10,000`) — each trade allocates a fixed dollar amount regardless of equity; useful for isolating per-trade P&L
- Both modes apply to every entry; the position size is recalculated at each trade open

#### 3. Cost Structure
- Exchange fee % (default: `0.10%`, step `0.01%`)
- Slippage % (default: `0.05%`, step `0.01%`)
- Applied on every trade entry **and** exit

#### 4. Walk-Forward Settings
- Training window: months slider (default: `12`, range `3–36`)
- Test window: months slider (default: `3`, range `1–12`)
- Step size: months (default: equals test window → non-overlapping folds)
  - Warning displayed if step < test window (overlapping test windows violate OOS independence)
  - **Fold windows are computed in actual trading days from the fetched data**, not fixed 21-day months. The engine derives the average trading days per month as `len(df) / ((df.index[-1] - df.index[0]).days / 365.25 * 12)` and uses this to convert month inputs to bar counts. This ensures fold boundaries align with real market calendars.
- **Optimization samples**: slider (default: `200`, range `50–1000`)
  - Controls how many random parameter combinations are tried per fold
  - Higher = more thorough but slower
- **Random seed**: number input (default: `42`)
  - Ensures reproducible results across runs; change to get a different random sample
- **Minimum trades filter**: number input (default: `5`)
  - Parameter sets producing fewer trades than this threshold are rejected during optimization
- **Minimum exposure filter**: % input (default: `5%`)
  - Parameter sets with market exposure below this threshold are rejected during optimization
  - Exposure = % of bars where position ≠ 0

#### 5. Strategy
- **Dropdown**: lists all discovered strategies by `STRATEGY_INFO["name"]`
- **Dynamic parameter controls**: rendered from `STRATEGY_INFO["parameters"]` and `STRATEGY_INFO["parameter_ranges"]`
  - One slider per parameter, pre-populated with `min`, `max`, and `step` from `parameter_ranges`
  - Default value set from `parameters`
  - Slider ranges define the search space for random search — the engine samples within these bounds
- **Strategy Cheat Sheet card**: displays `STRATEGY_INFO["description"]` and tags
- **Import Strategy**: file uploader accepting `.py` files, copies to `strategies/` folder after validation

#### Bottom of Sidebar
```
[ Run Backtest ]
```

---

## LLM Import Workflow (no AI in app)

The sidebar shows a permanent copyable prompt template:

> *"I need a Python strategy file for a trading app. The file must export `STRATEGY_INFO` (a dict with `name`, `description`, `parameters`, `parameter_ranges`, `warmup_bars`, `tags`) and `generate_signals(df, **params) -> pd.Series` returning 1 / 0 / -1. The `df` has daily OHLCV columns. No lookahead bias allowed. `parameter_ranges` must contain a `min`, `max`, and `step` for every key in `parameters`. `warmup_bars` must be a positive integer equal to the minimum number of bars needed before the indicators are reliable (e.g. 50 if your longest indicator period is 50). Do not account for Butterworth filter warm-up in `warmup_bars` — the engine adds that automatically. Here is my request: [describe your strategy or tweak]"*

**Workflow for tweaking an existing strategy:**
1. Open the existing `.py` file from the `strategies/` folder
2. Paste its contents + the prompt above into Claude / ChatGPT
3. Describe the tweak in plain English (e.g. *"Add a volume confirmation: only buy if today's volume is 20% above its 10-day average"*)
4. Paste the returned file back via the Import button

---

## Backtester — Walk-Forward Engine

### Fold Structure

```
indicator_warmup = STRATEGY_INFO["warmup_bars"]
filter_warmup    = 3 * filter_order              # e.g. 6 bars for a 2-pole filter
total_warmup     = indicator_warmup + filter_warmup

rng = np.random.default_rng(seed)   # seed from sidebar, default 42

for each fold:
    # --- Training window ---
    # Pad by total_warmup to cover both filter transients and indicator warm-up
    padded_train  = df[fold_start - total_warmup : fold_start + train_window]
    filtered_train = apply_butterworth(padded_train["Close"])
    padded_train["Close"] = filtered_train          # replace raw close with filtered
    usable_train  = padded_train.iloc[total_warmup:]  # discard warm-up rows

    # --- Test window ---
    padded_test   = df[fold_start + train_window - total_warmup : fold_start + train_window + test_window]
    filtered_test  = apply_butterworth(padded_test["Close"])
    padded_test["Close"] = filtered_test            # replace raw close with filtered
    usable_test   = padded_test.iloc[total_warmup:]   # discard warm-up rows

    best_params   = random_search(strategy, usable_train, n_samples, rng,
                                  min_trades=min_trades, min_exposure=min_exposure)
    # random_search rejects any parameter set where:
    #   trade_count < min_trades  OR  exposure < min_exposure
    # if ALL samples are rejected, fold is skipped and flagged in the UI

    # --- Signal execution — two-bar shift model ---
    # Signal generated at close of bar T
    # Execution assumed at open of bar T+1 (approximated as close of bar T)
    # Return captured: close T+1 → close T+2
    # shift(2) prevents capturing the pre-execution move that shift(1) would include.
    raw_signals = generate_signals(usable_test, **best_params)
    positions   = raw_signals.shift(2).fillna(0)   # two-bar lag
    raw_returns = usable_test["Close"].pct_change() * positions   # full-exposure return

    # Cost detection via diff()
    transitions = positions.diff().fillna(0)
    # transition != 0 → entry, exit, or reversal → apply fee + slippage

    # Force liquidation at last bar of test window
    # If position is non-zero at final bar: close it and charge exit cost

    oos_returns = raw_returns (with costs applied at transitions, sizing_fraction applied per rule 9)

    stitch oos_returns into final OOS equity curve

advance fold_start by step_size, repeat
```

> ⚠️ **Warm-up ratio warning**: If `total_warmup` is large relative to the training window (e.g. 56 bars of warm-up on a 3-month ≈ 63-bar training window), over 85% of training data is discarded before optimization. The UI must display a warning when `total_warmup / train_bars > 0.3` and suggest increasing the training window or reducing `warmup_bars`.

> ⚠️ **Remainder handling**: The last fold may not fill a full test window if `(data_end - first_test_start)` is not a perfect multiple of `step_bars`. The engine includes the partial fold if it contains at least `test_bars / 2` usable bars; otherwise it is silently dropped. The UI displays the actual fold count after the run.

### Optimization — Random Search

- **Target**: maximize **Sharpe Ratio** on the in-sample training window
- **Method**: random search — `n_samples` combinations are drawn using `np.random.default_rng(seed)` from the full parameter space
  - `n_samples` controlled by the **Optimization samples** sidebar slider (default: `200`)
  - `seed` controlled by the **Random seed** sidebar input (default: `42`) — same seed always produces identical results
  - Replaces exhaustive grid search, which would produce millions of combinations and take hours
  - At 200 samples × 10 folds = 2,000 runs total; completes in seconds on a typical laptop
- **Minimum trade filter**: parameter sets with fewer than `min_trades` trades on the training window are rejected (default: `5`)
- **Minimum exposure filter**: parameter sets where the strategy is invested less than `min_exposure`% of bars are rejected (default: `5%`)
- Both filters apply as AND conditions — a parameter set must pass both to be eligible
- If all `n_samples` are rejected in a fold, the fold is skipped and flagged with a warning in the UI
- Only the **out-of-sample (OOS)** returns are stitched into the final equity curve

### Cost Application

| Event | Cost |
|---|---|
| Entry | `fee% + slippage%` deducted from position |
| Exit | `fee% + slippage%` deducted from position |
| **Round-trip total** | `2 × (fee + slippage)` = **0.30%** at defaults |

### Butterworth Filter

- Applied to Close prices **before** signal generation, **per fold**
- Applied over the **padded** (warm-up buffered) data range using `total_warmup = indicator_warmup + filter_warmup`
- `indicator_warmup` comes from `STRATEGY_INFO["warmup_bars"]` and covers indicator transients only
- `filter_warmup` is computed by the engine as `3 × filter_order` (e.g. 6 bars for a 2-pole filter) and covers Butterworth transient ringing
- Both warm-up sources are stacked: `total_warmup` rows are discarded after filtering, before any indicator computation or signal generation
- Strategy authors set `warmup_bars` for their indicators only; they do not need to know about the filter
- Filter state is initialized fresh at the start of each fold — never applied across the full dataset
- 2-pole low-pass filter
- Cutoff frequency: configurable (default: `0.1` normalized)
- Implementation: **`scipy.signal.lfilter`** — strictly forward-only, never `filtfilt`
- `lfilter` introduces phase lag (signals fire slightly late relative to raw price) — this is the honest cost of causal filtering and is documented in the UI
- Purpose: smooth price noise with **zero lookahead bias**

### Signal Execution — Close-to-Close Model with Two-Bar Shift

The app uses a two-bar shift execution model:

```python
# Signal generated at close of bar T
# Execution assumed at open of bar T+1 (approximated as close of bar T)
# Return captured: close T+1 → close T+2
positions   = signals.shift(2).fillna(0)
raw_returns = close.pct_change() * positions   # full-exposure return, sizing applied separately
```

**Why two bars, not one**: `close.pct_change()` at bar T+1 equals `(close[T+1] - close[T]) / close[T]`. With a one-bar shift the position is active during this return, meaning the executor captures the full move from T to T+1 — including the portion that occurred before their assumed execution price at close T. This overstates performance. Shifting by two bars ensures the captured return starts *after* the execution bar: the position is active from close T+1 to close T+2, which is the honest first bar of exposure.

**Trade-off**: this adds one extra bar of lag compared to the one-bar shift model. Signals fire two bars after generation rather than one. This is documented in the UI so the user understands the assumption.

**Cost detection and force liquidation are unchanged** — `positions.diff()` still correctly identifies transitions regardless of shift size.

### Cost Detection via Position Diff

Costs are applied only when the position changes, detected using `positions.diff()`:

| `diff()` value | Meaning | Cost applied |
|---|---|---|
| `0 → 1` or `0 → -1` | Entry | fee + slippage |
| `1 → 0` or `-1 → 0` | Exit | fee + slippage |
| `1 → -1` or `-1 → 1` | Reversal | 2 × (fee + slippage) |
| `0` | No change | None |

### Fold Boundary — Force Liquidation

At the **last bar of every test window**, the engine forces the position to `0` regardless of the signal:

```python
# After computing positions = raw_signals.shift(2).fillna(0):
# The shift means the last two signal bars produce no positions (they are
# shifted beyond the end of the series). The engine must check the last
# POSITION value — not the last signal value — and close it if non-zero.

# cost_series is a pd.Series (same index as positions) initialised to 0.0,
# then populated by the positions.diff() cost detection pass before this block.
last_position = positions.iloc[-1]
if last_position != 0:
    positions.iloc[-1] = 0          # force flat
    # charge exit cost at the last bar
    cost_series.iloc[-1] += fee_pct + slippage_pct
```

- The exit cost is charged regardless of how small the remaining position is
- The next fold always starts completely flat with no inherited position
- This ensures costs are never hidden across fold boundaries and each fold is fully self-contained
- **Do not** check the last signal value as a proxy — with a two-bar shift the last two signals are never reflected in positions and will always read as 0, masking an open position carried from earlier bars

### Engine Output

```python
{
    "oos_equity":        pd.Series,   # stitched OOS equity curve in dollars
    "bh_equity":         pd.Series,   # buy & hold equity curve in dollars (cost-adjusted, same period)
    "folds":             list[dict],  # per-fold metadata: dates, best params, Sharpe
    "trade_log":         pd.DataFrame,# all trades with costs, dollar P&L, and fold_id column
    "metrics":           dict,        # summary statistics for OOS curve and buy & hold
    "param_stability":   dict         # per-parameter: values per fold, CV, median (suggested live value)
}
```

### Position Sizing & Dollar Equity

All equity curves are denominated in dollars starting from `initial_capital`.

`generate_signals` returns a direction (1, 0, -1). The `raw_returns` series is computed at full exposure:

```python
raw_returns = close.pct_change() * positions   # full-exposure daily return
```

The sizing fraction is then applied **before** the equity update, so partial allocations are correctly reflected:

```python
sizing_fraction = sizing_pct / 100.0           # e.g. 0.5 for 50% of equity
scaled_returns  = raw_returns * sizing_fraction
```

**% of equity mode** (default):
```python
# sizing_fraction from sidebar (default 1.0 = 100%)
equity[t] = equity[t-1] * (1 + scaled_returns[t] - cost[t] * sizing_fraction)
```
Position size compounds — gains increase future trade size, losses reduce it. At 100% sizing this is identical to `equity[t-1] * (1 + raw_returns[t] - cost[t])`.

**Fixed $ per trade mode**:
```python
# fixed_amount from sidebar (default $10,000); independent of current equity
trade_pnl  = fixed_amount * (raw_returns[t] - cost[t])
equity[t]  = equity[t-1] + trade_pnl
```
Each trade risks the same fixed dollar amount regardless of cumulative performance. The sizing fraction is not applied in this mode — `fixed_amount` is the absolute allocation per trade.

> ⚠️ Costs in `% of equity` mode are scaled by `sizing_fraction` because only the allocated portion of equity is exposed to entry/exit costs. At 50% sizing, only half the equity crosses the spread.

### Trading Day Calendar

Fold windows are specified in months by the user but converted to **actual trading day counts** from the fetched price data:

```python
trading_days_per_month = len(df) / ((df.index[-1] - df.index[0]).days / 365.25 * 12)
train_bars = round(train_months * trading_days_per_month)
test_bars  = round(test_months  * trading_days_per_month)
step_bars  = round(step_months  * trading_days_per_month)
```

This ensures fold boundaries align with the actual market calendar rather than assuming a fixed 21 trading days per month, which drifts materially over multi-year backtests.

### Buy & Hold Benchmark

Computed over the same date range as the OOS equity curve (i.e. from the end of the first training window, not the full dataset start). Costs are applied once at entry and once at exit using the same fee% + slippage% values set in the sidebar. This makes the comparison fair — the strategy pays round-trip costs on every trade, so buy & hold must too.

---

## Dashboard — Main Area

### Section 1: Walk-Forward Gantt Chart

- Plotly horizontal bar chart
- Each fold = two bars: **Blue** (training window) and **Orange** (blind test window)
- X-axis = calendar time, Y-axis = fold number
- `st.progress` bar updates live as folds complete

### Section 2: Metrics

| Metric | Walk-Forward OOS | Buy & Hold (cost-adjusted) |
|---|---|---|
| Initial Capital | | — |
| Final Capital | | |
| Net P&L ($) | | |
| Total Return (%) | | |
| Annual Return (%) | | |
| Sharpe Ratio | | |
| Max Drawdown (%) | | |
| Max Drawdown ($) | | |
| Exposure % | | 100% |
| Win Rate | | — |
| Total Trades | | 1 |
| Avg profit per trade ($) | | — |
| Avg loss per trade ($) | | — |

> Note: Sharpe Ratio uses risk-free rate = 0. Published Sharpe ratios often use T-bill rates, so values are not directly comparable.
> Exposure % = percentage of bars where the strategy holds a position (long or short). A strategy with low exposure but decent returns may be stronger than raw CAGR suggests.

### Section 2b: P&L Summary Cards

Four large stat cards displayed prominently above the equity chart:

| Net P&L | Annual Return | Max Drawdown | Sharpe Ratio |
|---|---|---|---|
| **+$3,240** | **8.4%** | **-12.3%** | **0.71** |

Colour coded: green if the strategy outperforms Buy & Hold on that metric, red if it underperforms.

### Section 3: Equity Curves (main chart)

- Large Plotly line chart with shared x-axis, Y-axis in dollars
- **Green line**: Walk-Forward OOS equity curve (starts at `initial_capital`)
- **Grey line**: Buy & Hold benchmark (cost-adjusted — one entry + one exit at the same fee% + slippage% as the strategy, starts at `initial_capital`)
- Both curves start from the same point: end of the first training window
- Hover tooltips show dollar value and % return from start on all lines
- Range slider at bottom

### Section 4: Drawdown Chart

- Plotly area chart directly below the equity curves
- Rolling drawdown plotted for both the OOS curve and Buy & Hold, using matching colours (green / grey)

### Section 5: Monthly Returns Heatmap

- Plotly heatmap — years as rows, months as columns
- Green = positive month, Red = negative month, Grey = no OOS data for that month
- Based on **OOS returns only**
- Months that span two folds are attributed to whichever fold contributes the majority of trading days in that month

### Section 6: Parameter Stability

This section answers: *"what parameters should I actually trade with going forward?"*

**Parameter stability table** — one row per fold, one column per optimized parameter. Rendered as a `st.dataframe` with conditional colour formatting: green = value close to the median, red = value far from the median. Allows the user to visually spot whether parameters are stable or jumping around across folds.

| Fold | rsi_period | sma_period | oversold | overbought |
|---|---|---|---|---|
| 1 | 12 | 25 | 28 | 65 |
| 2 | 14 | 20 | 30 | 70 |
| ... | | | | |

**Stability verdict** — displayed as a simple callout below the table:
- ✅ **Stable** — if the coefficient of variation (std / mean) across folds is < 0.2 for all parameters
- ⚠️ **Unstable** — if any parameter has CV ≥ 0.2; a warning is shown: *"Parameters vary significantly across folds. This strategy may not be robust enough to trade live."*

**Suggested live parameters** — the median value of each parameter across all folds, displayed as a copyable code block. This is the most statistically defensible choice for live trading.

```
Suggested parameters for live trading:
  rsi_period  : 13
  sma_period  : 22
  oversold    : 29
  overbought  : 68
```

> ⚠️ These are a starting point, not a guarantee. Past parameter stability does not ensure future stability.

---

### Section 7: Trade Log Table

Sortable `st.dataframe` with columns:

| Fold | Entry Date | Exit Date | Direction | Entry Price | Exit Price | Position Size ($) | Gross P&L ($) | Costs ($) | Net P&L ($) | Return (%) | Holding Days |
|---|---|---|---|---|---|---|---|---|---|---|---|

> Note: Costs ($) column shows the total round-trip cost (entry + exit fees and slippage) for each trade, enabling cost attribution audit.

---

## Metrics Definitions

| Metric | Formula |
|---|---|
| Final Capital | `initial_capital × (1 + total_return)` |
| Net P&L ($) | `final_capital - initial_capital` |
| Total Return | `(final_equity / initial_equity) - 1` |
| Annual Return | `(1 + total_return) ^ (trading_days_per_year / total_oos_bars) - 1` where `trading_days_per_year` is derived from actual data and `total_oos_bars` is the total number of bars in the stitched OOS equity curve |
| Sharpe Ratio | `mean(daily_returns) / std(daily_returns) × √252`, risk-free = 0. **Flat bars (position = 0) are included** as zero-return days in both mean and std. This is the same computation used inside random search during optimization, ensuring the reported Sharpe is directly comparable to the value being maximized. Excluding flat bars would produce a higher Sharpe that is inconsistent with what the optimizer selected for. |
| Max Drawdown (%) | `max((peak - trough) / peak)` over full period |
| Max Drawdown ($) | `max(peak_equity - trough_equity)` over full period |
| Exposure % | `bars where position ≠ 0 / total bars × 100` |
| Win Rate | `trades with net_pnl > 0 / total_trades` |
| Avg profit per trade | `mean(net_pnl) for winning trades` |
| Avg loss per trade | `mean(net_pnl) for losing trades` |

---

## Bundled Default Strategy — `rsi_sma.py`

- **Signal logic**: Go long when RSI(14) crosses above 30 **and** price is above SMA(20). Exit when RSI crosses above 70.
- **Direction**: Long-only
- **Optimization parameters and ranges**: `rsi_period` (default 14, range 5–30), `sma_period` (default 20, range 10–50), `oversold` (default 30, range 20–40), `overbought` (default 70, range 60–80) — all with step 1
- **warmup_bars**: `50` (covers the longest possible indicator period in the search range; filter warm-up is added automatically by the engine)
- **Tags**: `mean-reversion`, `long-only`, `trend-filter`

---

## Critical Implementation Rules

1. **`lfilter` only** — never `filtfilt`. The Butterworth filter must be causal with zero lookahead. `lfilter` introduces phase lag (signals fire slightly late) — this is expected and honest.

2. **Two-level warm-up: indicator + filter** — `total_warmup = STRATEGY_INFO["warmup_bars"] + (3 × filter_order)`. Strategy authors set `warmup_bars` for indicator transients only. The engine always adds `filter_warmup` on top. Both layers must be discarded before any indicator computation or signal generation. Never derive total lookback from `parameter_ranges` maxes — not all parameters are period lengths.

3. **Two-bar shift execution model** — `positions = signals.shift(2).fillna(0)`, `raw_returns = close.pct_change() * positions`. Signal at bar T is assumed to execute at open of bar T+1 (approximated as close T); return is captured from close T+1 to close T+2. `raw_returns` is at full exposure — sizing is applied separately per rule 9. This prevents capturing the pre-execution move that a one-bar shift would incorrectly include. Document this assumption in the UI.

4. **Cost detection via `diff()`** — use `positions.diff()` to identify entries, exits, and reversals. Apply fee + slippage only on transitions, not on every bar.

5. **Force liquidation at fold boundaries** — check the last *position* value (not the last signal value) at the end of every test window. If non-zero, set it to 0 and charge the exit cost. Never use the signal series as a proxy — with a two-bar shift the last two signal bars produce no positions and will always read as 0 even when a position is open. The next fold starts completely flat.

6. **Random search with fixed seed** — use `np.random.default_rng(seed)` with default seed `42`. Same seed = identical results across runs.

7. **Minimum trade and exposure filters** — reject any parameter set where `trade_count < min_trades` OR `exposure < min_exposure`. Both are AND conditions. If all samples in a fold are rejected, skip the fold and flag it in the UI.

8. **Initial capital is required** — all equity curves start at `initial_capital` (default `$10,000`). Never use normalised returns (starting at 1.0) for the final output — users read dollar values.

9. **Position sizing scales returns before equity update** — compute `raw_returns = close.pct_change() * positions` at full exposure, then multiply by `sizing_fraction` before the equity update. In `% of equity` mode: `equity[t] = equity[t-1] * (1 + raw_returns[t] * sizing_fraction - cost[t] * sizing_fraction)`. In `Fixed $` mode: `trade_pnl = fixed_amount * (raw_returns[t] - cost[t])`. Never apply `raw_returns` directly to the equity formula without scaling — this silently overstates P&L for any sizing below 100%.

10. **Fold windows in actual trading days** — convert month inputs using `trading_days_per_month = len(df) / ((df.index[-1] - df.index[0]).days / 365.25 * 12)`. Never assume 21 trading days per month.

11. **OOS-only equity curve** — the walk-forward result stitches test-window returns only.

12. **Costs on every trade** — fee + slippage applied at both entry and exit, no exceptions.

13. **Buy & Hold is cost-adjusted** — apply the same fee% + slippage% once at entry and once at exit. Buy & Hold starts at the same date as the first OOS test window.

14. **Parameter stability uses coefficient of variation** — CV = std / mean per parameter across folds. CV < 0.2 = stable, CV ≥ 0.2 = unstable. Suggested live parameters are the per-parameter median across all folds.

15. **Trade log includes fold_id and Costs ($)** — every trade row must reference which fold produced it and show the total round-trip cost as a separate column.

16. **No run-on-change** — backtest only triggers on button click, never on slider/input change.

17. **Strategy isolation** — each strategy file is a self-contained module; the engine calls only `generate_signals()` and reads only `STRATEGY_INFO`.

18. **Warm-up ratio warning** — display a UI warning when `total_warmup / train_bars > 0.3`. Suggest increasing the training window or reducing `warmup_bars`.

19. **Partial last fold** — include the final fold if it contains at least `test_bars / 2` usable bars; otherwise drop it silently. Display the actual fold count in the UI after the run.

20. **Monthly heatmap fold attribution** — months spanning two folds are attributed to the fold contributing the majority of trading days in that month.

21. **Filtered Close, raw everything else** — the engine replaces `df["Close"]` with the Butterworth-filtered series before calling `generate_signals`. `Open`, `High`, `Low`, and `Volume` are passed through unmodified. This is documented in the strategy contract docstring. Strategy authors must not use `High`/`Low` for range-based indicators (ATR, true range, Donchian channels) expecting them to be consistent with a filtered `Close` — they will not be. If a strategy requires range-based indicators, it should derive them from raw `Close` via `df["High"] - df["Low"]` only, not by mixing filtered `Close` with raw `High`/`Low`.

---

## `requirements.txt`

```
streamlit
plotly
yfinance
scipy
pandas
numpy
pandas-ta
```
