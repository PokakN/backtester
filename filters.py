import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

FILTER_ORDER = 2
FILTER_CUTOFF = 0.1
FILTER_WARMUP_BARS = 3 * FILTER_ORDER  # 6 bars


def build_butterworth(order: int = FILTER_ORDER, cutoff: float = FILTER_CUTOFF):
    """Return (b, a) Butterworth low-pass coefficients. Call once, reuse across folds."""
    return butter(order, cutoff, btype="low", analog=False)


def apply_filter(prices: pd.Series, b: np.ndarray, a: np.ndarray) -> pd.Series:
    """Apply causal lfilter to prices. Never uses filtfilt — zero lookahead."""
    nan_mask = prices.isna()
    filled = prices.ffill().bfill()
    filtered = lfilter(b, a, filled.values)
    result = pd.Series(filtered, index=prices.index)
    result[nan_mask] = np.nan
    return result
