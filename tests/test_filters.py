"""Tests for filters.py — Butterworth low-pass filter helpers."""
import numpy as np
import pandas as pd
import pytest
from filters import build_butterworth, apply_filter, FILTER_ORDER, FILTER_CUTOFF, FILTER_WARMUP_BARS


# ---------------------------------------------------------------------------
# build_butterworth
# ---------------------------------------------------------------------------

class TestBuildButterworth:
    def test_returns_two_arrays(self):
        result = build_butterworth()
        assert len(result) == 2

    def test_b_and_a_are_numpy_arrays(self):
        b, a = build_butterworth()
        assert isinstance(b, np.ndarray)
        assert isinstance(a, np.ndarray)

    def test_default_order_yields_correct_lengths(self):
        b, a = build_butterworth()
        # For a Butterworth of order N, b and a each have N+1 coefficients
        assert len(b) == FILTER_ORDER + 1
        assert len(a) == FILTER_ORDER + 1

    def test_custom_order(self):
        b, a = build_butterworth(order=4)
        assert len(b) == 5
        assert len(a) == 5

    def test_a0_coefficient_is_one(self):
        # scipy normalises so that a[0] == 1
        _, a = build_butterworth()
        assert a[0] == pytest.approx(1.0)

    def test_stable_filter_poles_inside_unit_circle(self):
        b, a = build_butterworth()
        poles = np.roots(a)
        assert all(abs(p) < 1.0 for p in poles)

    def test_warmup_bars_constant(self):
        assert FILTER_WARMUP_BARS == 3 * FILTER_ORDER


# ---------------------------------------------------------------------------
# apply_filter
# ---------------------------------------------------------------------------

class TestApplyFilter:
    @pytest.fixture
    def ba(self):
        return build_butterworth()

    def _make_prices(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        prices = 100 + np.cumsum(rng.standard_normal(n))
        return pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    def test_output_length_matches_input(self, ba):
        b, a = ba
        prices = self._make_prices(60)
        result = apply_filter(prices, b, a)
        assert len(result) == len(prices)

    def test_output_index_matches_input(self, ba):
        b, a = ba
        prices = self._make_prices(50)
        result = apply_filter(prices, b, a)
        assert result.index.equals(prices.index)

    def test_output_is_pandas_series(self, ba):
        b, a = ba
        prices = self._make_prices(30)
        result = apply_filter(prices, b, a)
        assert isinstance(result, pd.Series)

    def test_filtered_output_smoother_than_input(self, ba):
        # High-frequency noise should be attenuated
        b, a = ba
        n = 200
        t = np.arange(n)
        # Low-frequency trend + high-frequency noise
        prices = pd.Series(
            100 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 3),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        result = apply_filter(prices, b, a)
        input_std = float(prices.diff().dropna().std())
        output_std = float(result.diff().dropna().std())
        assert output_std < input_std

    def test_nan_positions_preserved(self, ba):
        b, a = ba
        prices = self._make_prices(50)
        prices.iloc[5] = np.nan
        prices.iloc[20] = np.nan
        result = apply_filter(prices, b, a)
        assert np.isnan(result.iloc[5])
        assert np.isnan(result.iloc[20])

    def test_non_nan_positions_not_corrupted(self, ba):
        b, a = ba
        prices = self._make_prices(50)
        prices.iloc[5] = np.nan
        result = apply_filter(prices, b, a)
        # All positions that were not NaN should have finite filtered values
        non_nan_mask = ~prices.isna()
        assert result[non_nan_mask].notna().all()

    def test_constant_series_passes_through_unchanged(self, ba):
        b, a = ba
        n = 80
        prices = pd.Series(
            [100.0] * n,
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        result = apply_filter(prices, b, a)
        # Check only the tail where the filter transient has fully decayed
        assert result.iloc[-20:].values == pytest.approx(100.0, abs=1e-3)

    def test_no_lookahead_causal_filter(self, ba):
        """Inserting a step at bar k must not affect filtered values before k."""
        b, a = ba
        prices_base = self._make_prices(80)
        prices_step = prices_base.copy()
        step_idx = 60
        prices_step.iloc[step_idx:] += 20.0

        result_base = apply_filter(prices_base, b, a)
        result_step = apply_filter(prices_step, b, a)

        # Before the step, filtered values must be identical
        np.testing.assert_array_equal(
            result_base.values[:step_idx],
            result_step.values[:step_idx],
        )
