import numpy as np
import pandas as pd
import logging
from filters import build_butterworth, apply_filter, FILTER_WARMUP_BARS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_trading_days_per_month(df: pd.DataFrame) -> float:
    """Return average trading days per calendar month for the given DataFrame."""
    days = (df.index[-1] - df.index[0]).days
    if days == 0:
        return 21.0
    return len(df) / (days / 365.25 * 12)


def sample_params(parameter_ranges: dict, rng: np.random.Generator) -> dict:
    """
    Sample one value per parameter from its [min, max] grid (inclusive, step-sized).
    All values cast to int (all current strategy params are integers).
    """
    params = {}
    for name, spec in parameter_ranges.items():
        lo, hi, step = spec["min"], spec["max"], spec["step"]
        n_steps = int(round((hi - lo) / step))
        choices = [lo + i * step for i in range(n_steps + 1)]
        params[name] = int(rng.choice(choices))
    return params


def evaluate_params(
    strategy_fn,
    df: pd.DataFrame,
    params: dict,
    min_trades: int,
    min_exposure_pct: float,
) -> "float | None":
    """
    Evaluate one parameter set on df.
    Returns Sharpe (float) or None if the set fails min-trades/exposure filters.
    No trading costs are applied here.
    """
    try:
        signals = strategy_fn(df, **params).astype(int)
    except Exception as exc:
        logger.warning("evaluate_params: strategy raised %s", exc)
        return None

    positions = signals.shift(2).fillna(0).astype(int)

    # Trade count: bars where position becomes non-zero (new entry or reversal)
    changed = (positions.diff().fillna(0) != 0) & (positions != 0)
    trade_count = int(changed.sum())

    exposure = (positions != 0).mean() * 100.0

    if trade_count < min_trades or exposure < min_exposure_pct:
        return None

    raw_returns = df["Close"].pct_change().fillna(0) * positions
    std = float(raw_returns.std(ddof=1))
    if std == 0:
        return 0.0
    sharpe = float(raw_returns.mean() / std * np.sqrt(252))
    return sharpe


# ---------------------------------------------------------------------------
# Trade log builder
# ---------------------------------------------------------------------------

def _build_trade_log(
    positions: pd.Series,
    usable_test: pd.DataFrame,
    cost_series: pd.Series,
    equity_series: pd.Series,
    sizing_mode: str,
    sizing_value: float,
    initial_equity: float,
) -> pd.DataFrame:
    """
    Walk the positions series and record each completed trade.

    Returns a DataFrame with columns:
        Entry Date, Exit Date, Direction, Entry Price, Exit Price,
        Gross P&L ($), Costs ($), Net P&L ($)
    """
    records = []
    idx = positions.index
    pos_values = positions.values
    n = len(pos_values)

    in_trade = False
    entry_bar = None  # integer position in array
    current_dir = 0

    def _record_trade(entry_bar_: int, exit_bar_: int):
        entry_idx_ = idx[entry_bar_]
        exit_idx_ = idx[exit_bar_]
        direction_ = current_dir
        entry_price_ = float(usable_test["Close"].loc[entry_idx_])
        exit_price_ = float(usable_test["Close"].loc[exit_idx_])

        if sizing_mode == "pct_equity":
            if entry_bar_ == 0:
                pos_size = initial_equity * sizing_value
            else:
                pos_size = float(equity_series.iloc[entry_bar_ - 1]) * sizing_value
        else:
            pos_size = sizing_value

        gross_pnl = pos_size * direction_ * (exit_price_ - entry_price_) / entry_price_
        costs_usd = float(cost_series.loc[entry_idx_:exit_idx_].sum()) * pos_size
        net_pnl = gross_pnl - costs_usd

        records.append({
            "Entry Date": entry_idx_,
            "Exit Date": exit_idx_,
            "Direction": "Long" if direction_ == 1 else "Short",
            "Entry Price": entry_price_,
            "Exit Price": exit_price_,
            "Gross P&L ($)": gross_pnl,
            "Costs ($)": costs_usd,
            "Net P&L ($)": net_pnl,
        })

    for t in range(n):
        p = int(pos_values[t])
        if not in_trade:
            if p != 0:
                in_trade = True
                entry_bar = t
                current_dir = p
        else:
            # sign change (reversal) or close to 0
            if p == 0:
                _record_trade(entry_bar, t)
                in_trade = False
                entry_bar = None
                current_dir = 0
            elif p != current_dir:
                # reversal: close old trade, open new one
                _record_trade(entry_bar, t)
                entry_bar = t
                current_dir = p

    # If still open at end (shouldn't happen after force-liquidation, but be safe)
    if in_trade and entry_bar is not None:
        _record_trade(entry_bar, n - 1)

    if not records:
        return pd.DataFrame(columns=[
            "Entry Date", "Exit Date", "Direction", "Entry Price", "Exit Price",
            "Gross P&L ($)", "Costs ($)", "Net P&L ($)",
        ])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def run_fold(
    df: pd.DataFrame,
    fold_start_idx: int,
    train_bars: int,
    test_bars: int,
    total_warmup: int,
    strategy_fn,
    parameter_ranges: dict,
    b: np.ndarray,
    a: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    min_trades: int,
    min_exposure_pct: float,
    fee_pct: float,
    slippage_pct: float,
    sizing_mode: str,
    sizing_value: float,
    initial_equity: float,
    fold_id: int,
) -> "dict | None":
    """
    Run one walk-forward fold.

    Returns a fold dict or None if the fold cannot be computed.
    """
    # ------------------------------------------------------------------
    # 1. Build padded training window
    # ------------------------------------------------------------------
    pad_start = max(0, fold_start_idx - total_warmup)
    train_end = fold_start_idx + train_bars
    train_padded = df.iloc[pad_start:train_end].copy()

    filtered_close_train = apply_filter(train_padded["Close"], b, a)
    train_padded["Close"] = filtered_close_train

    # Discard the warmup padding rows
    rows_to_discard = fold_start_idx - pad_start  # actual warmup rows prepended
    if rows_to_discard >= len(train_padded):
        logger.warning("Fold %d: not enough rows after padding; skipping.", fold_id)
        return None
    usable_train = train_padded.iloc[rows_to_discard:].copy()

    # ------------------------------------------------------------------
    # 2. Build padded test window
    # ------------------------------------------------------------------
    test_pad_start = max(0, fold_start_idx + train_bars - total_warmup)
    test_end = fold_start_idx + train_bars + test_bars
    test_padded = df.iloc[test_pad_start:test_end].copy()

    filtered_close_test = apply_filter(test_padded["Close"], b, a)
    test_padded["Close"] = filtered_close_test

    test_rows_to_discard = (fold_start_idx + train_bars) - test_pad_start
    if test_rows_to_discard >= len(test_padded):
        logger.warning("Fold %d: not enough test rows after padding; skipping.", fold_id)
        return None
    usable_test = test_padded.iloc[test_rows_to_discard:].copy()

    if len(usable_test) == 0:
        return None

    # ------------------------------------------------------------------
    # 3. Random search on in-sample (usable_train)
    # ------------------------------------------------------------------
    best_sharpe = None
    best_params = None

    for _ in range(n_samples):
        params = sample_params(parameter_ranges, rng)
        sharpe = evaluate_params(
            strategy_fn, usable_train, params, min_trades, min_exposure_pct
        )
        if sharpe is None:
            continue
        if best_sharpe is None or sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params

    if best_params is None:
        logger.warning("Fold %d: all %d samples rejected; skipping.", fold_id, n_samples)
        return {
            "fold_id": fold_id,
            "train_start": usable_train.index[0] if len(usable_train) else None,
            "train_end": usable_train.index[-1] if len(usable_train) else None,
            "test_start": usable_test.index[0],
            "test_end": usable_test.index[-1],
            "equity": None,
            "trades": pd.DataFrame(),
            "best_params": None,
            "in_sample_sharpe": None,
            "skipped": True,
        }

    # ------------------------------------------------------------------
    # 4. Out-of-sample: generate signals on test set
    # ------------------------------------------------------------------
    signals = strategy_fn(usable_test, **best_params).astype(int)
    positions = signals.shift(2).fillna(0).astype(int).copy()

    # Force liquidation: last bar position must be 0
    if positions.iloc[-1] != 0:
        positions.iloc[-1] = 0

    # ------------------------------------------------------------------
    # 5. Cost series
    # ------------------------------------------------------------------
    transitions = positions.diff().fillna(0).abs()
    cost_series = transitions * (fee_pct + slippage_pct)

    # ------------------------------------------------------------------
    # 6. Raw returns
    # ------------------------------------------------------------------
    raw_returns = usable_test["Close"].pct_change().fillna(0) * positions

    # ------------------------------------------------------------------
    # 7. Build equity curve
    # ------------------------------------------------------------------
    n = len(positions)
    equity = np.zeros(n)
    equity[0] = initial_equity

    if sizing_mode == "pct_equity":
        sf = sizing_value
        for t in range(1, n):
            equity[t] = equity[t - 1] * (
                1 + raw_returns.iloc[t] * sf - cost_series.iloc[t] * sf
            )
    else:
        fixed_amount = sizing_value
        for t in range(1, n):
            equity[t] = equity[t - 1] + fixed_amount * (
                raw_returns.iloc[t] - cost_series.iloc[t]
            )

    equity_series = pd.Series(equity, index=positions.index)

    # ------------------------------------------------------------------
    # 8. Trade log
    # ------------------------------------------------------------------
    trades_df = _build_trade_log(
        positions,
        usable_test,
        cost_series,
        equity_series,
        sizing_mode,
        sizing_value,
        initial_equity,
    )

    # ------------------------------------------------------------------
    # 9. In-sample Sharpe (for reporting only)
    # ------------------------------------------------------------------
    is_signals = strategy_fn(usable_train, **best_params).astype(int)
    is_positions = is_signals.shift(2).fillna(0).astype(int)
    is_returns = usable_train["Close"].pct_change().fillna(0) * is_positions
    is_std = float(is_returns.std(ddof=1))
    in_sample_sharpe = (
        float(is_returns.mean() / is_std * np.sqrt(252)) if is_std != 0 else 0.0
    )

    return {
        "fold_id": fold_id,
        "train_start": usable_train.index[0],
        "train_end": usable_train.index[-1],
        "test_start": usable_test.index[0],
        "test_end": usable_test.index[-1],
        "equity": equity_series,
        "trades": trades_df,
        "best_params": best_params,
        "in_sample_sharpe": in_sample_sharpe,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_metrics(equity: pd.Series, initial: float, trades_df: pd.DataFrame) -> dict:
    """Compute standard performance metrics for an equity curve."""
    final = float(equity.iloc[-1])
    n_days = (equity.index[-1] - equity.index[0]).days
    if n_days <= 0:
        n_days = 1

    total_return_pct = (final - initial) / initial * 100.0
    annual_return_pct = ((final / initial) ** (252.0 / n_days) - 1) * 100.0

    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    max_dd_pct = float(drawdowns.min()) * 100.0

    daily_returns = equity.pct_change().fillna(0)
    dd_std = float(daily_returns.std(ddof=1))
    sharpe = float(daily_returns.mean() / dd_std * np.sqrt(252)) if dd_std != 0 else 0.0

    # Win rate and avg trade %
    win_rate_pct = np.nan
    avg_trade_pct = np.nan
    if not trades_df.empty and "Net P&L ($)" in trades_df.columns:
        wins = (trades_df["Net P&L ($)"] > 0).sum()
        win_rate_pct = wins / len(trades_df) * 100.0
        if "Entry Price" in trades_df.columns and "Exit Price" in trades_df.columns and "Direction" in trades_df.columns:
            direction_sign = trades_df["Direction"].map({"Long": 1, "Short": -1}).fillna(1)
            trade_pcts = direction_sign * (trades_df["Exit Price"] - trades_df["Entry Price"]) / trades_df["Entry Price"] * 100.0
            avg_trade_pct = float(trade_pcts.mean())

    return {
        "total_return_pct": total_return_pct,
        "annual_return_pct": annual_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "sharpe": sharpe,
        "win_rate_pct": win_rate_pct,
        "avg_trade_pct": avg_trade_pct,
        "final_equity": final,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    price_field: str,
    df: pd.DataFrame,
    strategy_name: str,
    strategy_fn,
    parameter_ranges: dict,
    warmup_bars: int,
    train_months: int,
    test_months: int,
    step_months: float,
    n_samples: int,
    seed: int = 42,
    min_trades: int = 10,
    min_exposure_pct: float = 5.0,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.001,
    sizing_mode: str = "pct_equity",
    sizing_value: float = 1.0,
    initial_capital: float = 10_000.0,
    progress_callback=None,
) -> dict:
    """
    Run a full walk-forward backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and a "Close" column (OHLCV).
        The raw Close column will be preserved as "_raw_close" before any
        filtering so B&H can use unfiltered prices.

    Returns
    -------
    dict with keys: folds, oos_equity, bh_equity, metrics, trades,
                    param_stability, warnings
    """
    if df.empty or "Close" not in df.columns:
        raise ValueError("df must be non-empty with a 'Close' column")

    warnings_list: list[str] = []

    # -----------------------------------------------------------------------
    # Store raw (pre-filter) close for B&H
    # -----------------------------------------------------------------------
    df = df.copy()
    df["_raw_close"] = df["Close"].copy()

    # -----------------------------------------------------------------------
    # Build filter coefficients once
    # -----------------------------------------------------------------------
    b, a = build_butterworth()

    # -----------------------------------------------------------------------
    # Convert months → bars
    # -----------------------------------------------------------------------
    tdpm = compute_trading_days_per_month(df)
    train_bars = int(round(train_months * tdpm))
    test_bars = int(round(test_months * tdpm))
    step_bars = int(round(step_months * tdpm))

    filter_warmup = FILTER_WARMUP_BARS
    total_warmup = warmup_bars + filter_warmup

    if total_warmup / train_bars > 0.3:
        msg = (
            f"Warm-up ({total_warmup} bars) is {total_warmup/train_bars:.0%} of "
            f"train window ({train_bars} bars). Consider longer train_months."
        )
        warnings_list.append(msg)
        logger.warning(msg)

    if step_months < test_months:
        msg = (
            "step_months < test_months: walk-forward windows overlap. "
            "OOS periods will not be fully independent."
        )
        warnings_list.append(msg)

    # -----------------------------------------------------------------------
    # Generate fold start indices
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    fold_starts = []
    idx = 0
    while True:
        if idx + train_bars + test_bars <= len(df):
            fold_starts.append(idx)
        else:
            remaining = len(df) - idx - train_bars
            if remaining >= test_bars // 2:
                fold_starts.append(idx)
            break
        idx += step_bars

    total_folds = len(fold_starts)

    if total_folds == 0:
        raise ValueError(
            "Not enough data for even one fold. "
            f"Need at least {train_bars + test_bars} bars, have {len(df)}."
        )

    # -----------------------------------------------------------------------
    # Run folds
    # -----------------------------------------------------------------------
    folds = []
    last_oos_equity_end = initial_capital

    for fold_i, fold_start_idx in enumerate(fold_starts):
        if progress_callback is not None:
            try:
                progress_callback(fold_i, total_folds, f"Fold {fold_i + 1}/{total_folds}")
            except Exception:
                pass

        fold_result = run_fold(
            df=df,
            fold_start_idx=fold_start_idx,
            train_bars=train_bars,
            test_bars=test_bars,
            total_warmup=total_warmup,
            strategy_fn=strategy_fn,
            parameter_ranges=parameter_ranges,
            b=b,
            a=a,
            n_samples=n_samples,
            rng=rng,
            min_trades=min_trades,
            min_exposure_pct=min_exposure_pct,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
            sizing_mode=sizing_mode,
            sizing_value=sizing_value,
            initial_equity=last_oos_equity_end,
            fold_id=fold_i + 1,
        )

        if fold_result is None:
            continue

        if not fold_result["skipped"] and fold_result["equity"] is not None:
            last_oos_equity_end = float(fold_result["equity"].iloc[-1])

        folds.append(fold_result)

    if progress_callback is not None:
        try:
            progress_callback(total_folds, total_folds, "Done")
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Stitch equity curve and trades
    # -----------------------------------------------------------------------
    valid_folds = [f for f in folds if not f["skipped"] and f["equity"] is not None]

    if not valid_folds:
        raise ValueError("All folds were skipped. Try looser filters or more data.")

    oos_equity = pd.concat([f["equity"] for f in valid_folds])
    all_trades = pd.concat(
        [f["trades"] for f in valid_folds if not f["trades"].empty],
        ignore_index=True,
    ) if any(not f["trades"].empty for f in valid_folds) else pd.DataFrame(
        columns=["Entry Date", "Exit Date", "Direction", "Entry Price", "Exit Price",
                 "Gross P&L ($)", "Costs ($)", "Net P&L ($)"]
    )

    # -----------------------------------------------------------------------
    # Buy & Hold benchmark (uses raw unfiltered close)
    # -----------------------------------------------------------------------
    oos_start = valid_folds[0]["test_start"]
    oos_end = valid_folds[-1]["test_end"]

    bh_prices = df.loc[oos_start:oos_end, "_raw_close"]
    entry_cost = fee_pct + slippage_pct
    exit_cost = fee_pct + slippage_pct

    bh_equity = initial_capital * (bh_prices / bh_prices.iloc[0]) * (1 - entry_cost)
    bh_equity = bh_equity.copy()
    bh_equity.iloc[-1] *= (1 - exit_cost)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    oos_metrics = _compute_metrics(oos_equity, initial_capital, all_trades)
    bh_metrics = _compute_metrics(bh_equity, initial_capital, pd.DataFrame())

    metrics = {"oos": oos_metrics, "bh": bh_metrics}

    # -----------------------------------------------------------------------
    # Parameter stability
    # -----------------------------------------------------------------------
    param_stability: dict = {}
    if valid_folds and valid_folds[0]["best_params"]:
        param_names = list(valid_folds[0]["best_params"].keys())
        for param in param_names:
            values = [
                f["best_params"][param]
                for f in valid_folds
                if f["best_params"] and param in f["best_params"]
            ]
            if not values:
                continue
            mean_val = float(np.mean(values))
            cv = float(np.std(values, ddof=1) / mean_val) if mean_val != 0 else 0.0
            median_val = float(np.median(values))
            param_stability[param] = {
                "values": values,
                "cv": cv,
                "median": median_val,
                "stable": cv < 0.2,
            }

    return {
        "folds": folds,
        "oos_equity": oos_equity,
        "bh_equity": bh_equity,
        "metrics": metrics,
        "trades": all_trades,
        "param_stability": param_stability,
        "warnings": warnings_list,
    }
