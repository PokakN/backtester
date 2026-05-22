import numpy as np
import pandas as pd
import pandas_ta as ta

STRATEGY_INFO = {
    "name": "RSI + SMA Filter",
    "description": (
        "Goes long when RSI crosses above the oversold level and price is above SMA. "
        "Exits when RSI crosses above the overbought level. Long-only."
    ),
    "parameters": {
        "rsi_period": 14,
        "sma_period": 20,
        "oversold": 30,
        "overbought": 70,
    },
    "parameter_ranges": {
        "rsi_period": {"min": 5,  "max": 30, "step": 1},
        "sma_period": {"min": 10, "max": 50, "step": 1},
        "oversold":   {"min": 20, "max": 40, "step": 1},
        "overbought": {"min": 60, "max": 80, "step": 1},
    },
    "warmup_bars": 50,
    "tags": ["mean-reversion", "long-only", "trend-filter"],
}


def generate_signals(df: pd.DataFrame, rsi_period: int, sma_period: int,
                     oversold: int, overbought: int) -> pd.Series:
    """
    df guaranteed columns: Open, High, Low, Close, Volume (daily OHLCV)
    Close is Butterworth-filtered; all other columns are raw.

    Returns pd.Series of int (1=long, 0=flat, -1=short), same index as df.
    No lookahead — only uses data available at row i.
    """
    close = df["Close"]

    rsi = ta.rsi(close, length=rsi_period)
    sma = ta.sma(close, length=sma_period)

    rsi_prev = rsi.shift(1)

    entry = (rsi_prev <= oversold) & (rsi > oversold) & (close > sma)
    exit_sig = (rsi_prev <= overbought) & (rsi > overbought)

    signal = pd.Series(np.nan, index=df.index, dtype=float)
    signal[entry] = 1.0
    signal[exit_sig] = 0.0

    signal = signal.ffill().fillna(0.0).clip(0.0, 1.0).astype(int)
    return signal
