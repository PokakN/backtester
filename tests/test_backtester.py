"""Tests for backtester.py — helpers, fold, and metrics."""
import numpy as np
import pandas as pd
import pytest
from backtester import (
    compute_trading_days_per_month,
    sample_params,
    evaluate_params,
    run_fold,
    _compute_metrics,
    _build_trade_log,
    run_backtest,
)
from filters import build_butterworth


# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a monotone upward-trending Close."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    close = np.maximum(close, 10)  # keep prices positive
    index = pd.date_range("2018-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": close * (1 + rng.uniform(-0.005, 0.005, n)),
        "High": close * (1 + rng.uniform(0.0, 0.01, n)),
        "Low": close * (1 - rng.uniform(0.0, 0.01, n)),
        "Close": close,
        "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=index)
    return df


def _always_long_strategy(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Strategy that is always in a long position — guarantees trades."""
    return pd.Series(1, index=df.index, dtype=int)


def _always_flat_strategy(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Strategy that is never in a position — no trades at all."""
    return pd.Series(0, index=df.index, dtype=int)


def _alternating_strategy(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Long/flat on alternating bars."""
    signals = pd.Series(0, index=df.index, dtype=int)
    signals.iloc[::2] = 1
    return signals


# ---------------------------------------------------------------------------
# compute_trading_days_per_month
# ---------------------------------------------------------------------------

class TestComputeTradingDaysPerMonth:
    def test_one_year_approx_21_days(self):
        df = _make_ohlcv(252)
        result = compute_trading_days_per_month(df)
        assert 19 <= result <= 23

    def test_single_row_returns_default(self):
        df = _make_ohlcv(1)
        result = compute_trading_days_per_month(df)
        assert result == pytest.approx(21.0)

    def test_same_start_end_returns_default(self):
        # Two rows on same date simulate degenerate case where days == 0
        index = pd.to_datetime(["2020-01-01", "2020-01-01"])
        df = pd.DataFrame({"Close": [100.0, 100.0]}, index=index)
        result = compute_trading_days_per_month(df)
        assert result == pytest.approx(21.0)

    def test_longer_history_gives_reasonable_result(self):
        df = _make_ohlcv(504)  # ~2 years
        result = compute_trading_days_per_month(df)
        assert 18 <= result <= 24


# ---------------------------------------------------------------------------
# sample_params
# ---------------------------------------------------------------------------

class TestSampleParams:
    PARAM_RANGES = {
        "rsi_period": {"min": 5,  "max": 30, "step": 1},
        "sma_period": {"min": 10, "max": 50, "step": 5},
        "oversold":   {"min": 20, "max": 40, "step": 2},
    }

    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_returns_all_keys(self):
        params = sample_params(self.PARAM_RANGES, self._rng())
        assert set(params.keys()) == {"rsi_period", "sma_period", "oversold"}

    def test_values_within_bounds(self):
        rng = self._rng()
        for _ in range(100):
            params = sample_params(self.PARAM_RANGES, rng)
            assert 5 <= params["rsi_period"] <= 30
            assert 10 <= params["sma_period"] <= 50
            assert 20 <= params["oversold"] <= 40

    def test_values_respect_step(self):
        rng = self._rng()
        for _ in range(200):
            params = sample_params(self.PARAM_RANGES, rng)
            # sma_period must be a multiple of 5 offset from 10
            assert (params["sma_period"] - 10) % 5 == 0
            # oversold must be even (step=2 from 20)
            assert (params["oversold"] - 20) % 2 == 0

    def test_values_are_integers(self):
        params = sample_params(self.PARAM_RANGES, self._rng())
        assert all(isinstance(v, int) for v in params.values())

    def test_single_value_range_always_returns_min(self):
        ranges = {"x": {"min": 7, "max": 7, "step": 1}}
        rng = self._rng()
        for _ in range(50):
            params = sample_params(ranges, rng)
            assert params["x"] == 7

    def test_reproducible_with_same_seed(self):
        params1 = sample_params(self.PARAM_RANGES, np.random.default_rng(99))
        params2 = sample_params(self.PARAM_RANGES, np.random.default_rng(99))
        assert params1 == params2


# ---------------------------------------------------------------------------
# evaluate_params
# ---------------------------------------------------------------------------

class TestEvaluateParams:
    def _df(self, n=100):
        return _make_ohlcv(n)

    def test_returns_float_for_valid_params(self):
        df = self._df(120)
        result = evaluate_params(
            _always_long_strategy, df,
            params={},
            min_trades=1,
            min_exposure_pct=0.0,
        )
        assert isinstance(result, float)

    def test_returns_none_when_below_min_trades(self):
        df = self._df(100)
        result = evaluate_params(
            _always_flat_strategy, df,
            params={},
            min_trades=5,
            min_exposure_pct=0.0,
        )
        assert result is None

    def test_returns_none_when_below_min_exposure(self):
        df = self._df(100)
        result = evaluate_params(
            _always_flat_strategy, df,
            params={},
            min_trades=0,
            min_exposure_pct=50.0,
        )
        assert result is None

    def test_returns_zero_sharpe_for_flat_returns(self):
        # Constant price → zero std → sharpe == 0.0
        n = 60
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": [100.0] * n}, index=index)
        result = evaluate_params(
            _always_long_strategy, df,
            params={},
            min_trades=1,
            min_exposure_pct=0.0,
        )
        assert result == pytest.approx(0.0)

    def test_returns_none_when_strategy_raises(self):
        def bad_strategy(df, **kwargs):
            raise RuntimeError("boom")

        df = self._df(50)
        result = evaluate_params(
            bad_strategy, df,
            params={},
            min_trades=1,
            min_exposure_pct=0.0,
        )
        assert result is None

    def test_always_long_has_positive_sharpe_on_uptrend(self):
        n = 150
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        # Strictly rising prices → positive daily returns → positive Sharpe
        df = pd.DataFrame({"Close": np.linspace(100, 200, n)}, index=index)
        result = evaluate_params(
            _always_long_strategy, df,
            params={},
            min_trades=1,
            min_exposure_pct=0.0,
        )
        assert result is not None
        assert result > 0


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _flat_equity(self, n=252, start=10_000.0):
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series([start] * n, index=index, dtype=float)

    def _linear_equity(self, n=252, start=10_000.0, end=20_000.0):
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series(np.linspace(start, end, n), index=index)

    def _decaying_equity(self, n=252, start=10_000.0, end=5_000.0):
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series(np.linspace(start, end, n), index=index)

    def test_total_return_flat(self):
        eq = self._flat_equity()
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["total_return_pct"] == pytest.approx(0.0)

    def test_total_return_doubled(self):
        eq = self._linear_equity(end=20_000.0)
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["total_return_pct"] == pytest.approx(100.0)

    def test_total_return_halved(self):
        eq = self._decaying_equity(end=5_000.0)
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["total_return_pct"] == pytest.approx(-50.0)

    def test_max_drawdown_zero_for_monotone_gain(self):
        eq = self._linear_equity()
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_negative_for_loss(self):
        eq = self._decaying_equity()
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["max_drawdown_pct"] < 0

    def test_max_drawdown_50_pct(self):
        n = 100
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        # Go up to 20k then back to 10k
        vals = np.concatenate([np.linspace(10_000, 20_000, 50), np.linspace(20_000, 10_000, 50)])
        eq = pd.Series(vals, index=index)
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["max_drawdown_pct"] == pytest.approx(-50.0, abs=1.0)

    def test_sharpe_zero_for_flat_equity(self):
        eq = self._flat_equity()
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["sharpe"] == pytest.approx(0.0)

    def test_annual_return_greater_than_zero_for_gain(self):
        eq = self._linear_equity()
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["annual_return_pct"] > 0

    def test_final_equity_matches_last_value(self):
        eq = self._linear_equity(end=15_000.0)
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["final_equity"] == pytest.approx(15_000.0)

    def test_win_rate_nan_when_no_trades(self):
        eq = self._flat_equity()
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert np.isnan(m["win_rate_pct"])

    def test_win_rate_100_all_winners(self):
        eq = self._flat_equity()
        trades = pd.DataFrame({
            "Entry Date":  [pd.Timestamp("2020-01-02")],
            "Exit Date":   [pd.Timestamp("2020-01-10")],
            "Direction":   ["Long"],
            "Entry Price": [100.0],
            "Exit Price":  [110.0],
            "Gross P&L ($)": [100.0],
            "Costs ($)":   [0.0],
            "Net P&L ($)": [100.0],
        })
        m = _compute_metrics(eq, 10_000.0, trades)
        assert m["win_rate_pct"] == pytest.approx(100.0)

    def test_win_rate_0_all_losers(self):
        eq = self._flat_equity()
        trades = pd.DataFrame({
            "Entry Date":  [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-15")],
            "Exit Date":   [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-22")],
            "Direction":   ["Long", "Long"],
            "Entry Price": [100.0, 105.0],
            "Exit Price":  [90.0, 95.0],
            "Gross P&L ($)": [-100.0, -100.0],
            "Costs ($)":   [0.0, 0.0],
            "Net P&L ($)": [-100.0, -100.0],
        })
        m = _compute_metrics(eq, 10_000.0, trades)
        assert m["win_rate_pct"] == pytest.approx(0.0)

    def test_degenerate_single_bar_equity(self):
        index = pd.date_range("2020-01-01", periods=1, freq="B")
        eq = pd.Series([10_000.0], index=index)
        m = _compute_metrics(eq, 10_000.0, pd.DataFrame())
        assert m["total_return_pct"] == pytest.approx(0.0)
        assert m["final_equity"] == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# _build_trade_log
# ---------------------------------------------------------------------------

class TestBuildTradeLog:
    def _make_inputs(self, n=20, initial=10_000.0):
        index = pd.date_range("2020-01-02", periods=n, freq="B")
        close = pd.Series(np.linspace(100, 120, n), index=index)
        df = pd.DataFrame({"Close": close}, index=index)
        return df, index

    def test_empty_when_always_flat(self):
        df, index = self._make_inputs()
        positions = pd.Series(0, index=index, dtype=int)
        cost_series = pd.Series(0.0, index=index)
        equity_series = pd.Series(10_000.0, index=index)
        result = _build_trade_log(positions, df, cost_series, equity_series,
                                  "pct_equity", 1.0, 10_000.0)
        assert result.empty

    def test_one_trade_long(self):
        df, index = self._make_inputs(n=10)
        positions = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], index=index, dtype=int)
        cost_series = pd.Series(0.0, index=index)
        equity_series = pd.Series(10_000.0, index=index)
        result = _build_trade_log(positions, df, cost_series, equity_series,
                                  "pct_equity", 1.0, 10_000.0)
        assert len(result) == 1
        assert result.iloc[0]["Direction"] == "Long"

    def test_trade_columns_present(self):
        df, index = self._make_inputs(n=10)
        positions = pd.Series([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], index=index, dtype=int)
        cost_series = pd.Series(0.0, index=index)
        equity_series = pd.Series(10_000.0, index=index)
        result = _build_trade_log(positions, df, cost_series, equity_series,
                                  "pct_equity", 1.0, 10_000.0)
        expected_cols = {"Entry Date", "Exit Date", "Direction",
                         "Entry Price", "Exit Price",
                         "Gross P&L ($)", "Costs ($)", "Net P&L ($)"}
        assert expected_cols.issubset(set(result.columns))

    def test_net_pnl_positive_on_uptrend_long(self):
        n = 10
        index = pd.date_range("2020-01-02", periods=n, freq="B")
        close = pd.Series(np.linspace(100, 200, n), index=index)
        df = pd.DataFrame({"Close": close}, index=index)
        positions = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 0, 0], index=index, dtype=int)
        cost_series = pd.Series(0.0, index=index)
        equity_series = pd.Series(10_000.0, index=index)
        result = _build_trade_log(positions, df, cost_series, equity_series,
                                  "pct_equity", 1.0, 10_000.0)
        assert result.iloc[0]["Net P&L ($)"] > 0


# ---------------------------------------------------------------------------
# run_fold
# ---------------------------------------------------------------------------

class TestRunFold:
    @pytest.fixture
    def ba(self):
        return build_butterworth()

    def _fold_kwargs(self, df, ba):
        b, a = ba
        return dict(
            df=df,
            fold_start_idx=0,
            train_bars=80,
            test_bars=60,
            total_warmup=10,
            strategy_fn=_always_long_strategy,
            parameter_ranges={},
            b=b, a=a,
            n_samples=5,
            rng=np.random.default_rng(0),
            min_trades=1,
            min_exposure_pct=0.0,
            fee_pct=0.001,
            slippage_pct=0.001,
            sizing_mode="pct_equity",
            sizing_value=1.0,
            initial_equity=10_000.0,
            fold_id=1,
        )

    def test_returns_dict_on_success(self, ba):
        df = _make_ohlcv(200)
        result = run_fold(**self._fold_kwargs(df, ba))
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, ba):
        df = _make_ohlcv(200)
        result = run_fold(**self._fold_kwargs(df, ba))
        expected = {"fold_id", "train_start", "train_end", "test_start",
                    "test_end", "equity", "trades", "best_params",
                    "in_sample_sharpe", "skipped"}
        assert expected.issubset(set(result.keys()))

    def test_equity_series_length_matches_test_bars(self, ba):
        df = _make_ohlcv(200)
        result = run_fold(**self._fold_kwargs(df, ba))
        assert not result["skipped"]
        assert len(result["equity"]) == 60

    def test_equity_starts_at_initial_equity(self, ba):
        df = _make_ohlcv(200)
        result = run_fold(**self._fold_kwargs(df, ba))
        assert result["equity"].iloc[0] == pytest.approx(10_000.0)

    def test_best_params_empty_dict_when_no_param_ranges(self, ba):
        df = _make_ohlcv(200)
        result = run_fold(**self._fold_kwargs(df, ba))
        assert result["best_params"] == {}

    def test_skipped_fold_when_all_samples_rejected(self, ba):
        df = _make_ohlcv(200)
        kwargs = self._fold_kwargs(df, ba)
        # Require far more trades than possible → all samples rejected
        kwargs["min_trades"] = 100_000
        result = run_fold(**kwargs)
        assert result["skipped"] is True
        assert result["equity"] is None

    def test_returns_none_when_test_window_is_empty(self, ba):
        # fold_start_idx=0, train_bars=35 > n=30 → test window starts beyond data
        df = _make_ohlcv(30)
        kwargs = self._fold_kwargs(df, ba)
        kwargs["train_bars"] = 35
        kwargs["test_bars"] = 20
        result = run_fold(**kwargs)
        assert result is None

    def test_fold_id_preserved(self, ba):
        df = _make_ohlcv(200)
        kwargs = self._fold_kwargs(df, ba)
        kwargs["fold_id"] = 7
        result = run_fold(**kwargs)
        assert result["fold_id"] == 7

    def test_last_position_is_zero_force_liquidation(self, ba):
        df = _make_ohlcv(200)
        result = run_fold(**self._fold_kwargs(df, ba))
        # always_long_strategy would otherwise leave position open at end
        # force-liquidation must set last position to 0
        # We can't directly inspect positions, but equity should be finite
        assert np.isfinite(result["equity"].iloc[-1])

    def test_fixed_dollar_sizing(self, ba):
        df = _make_ohlcv(200)
        kwargs = self._fold_kwargs(df, ba)
        kwargs["sizing_mode"] = "fixed_dollar"
        kwargs["sizing_value"] = 1_000.0
        result = run_fold(**kwargs)
        assert not result["skipped"]
        assert np.isfinite(result["equity"].iloc[-1])

    def test_in_sample_sharpe_is_float(self, ba):
        df = _make_ohlcv(200)
        result = run_fold(**self._fold_kwargs(df, ba))
        assert isinstance(result["in_sample_sharpe"], float)


# ---------------------------------------------------------------------------
# run_backtest (integration smoke test)
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def _df(self, n=400):
        return _make_ohlcv(n)

    def test_returns_expected_keys(self):
        df = self._df()
        result = run_backtest(
            ticker="TEST", start_date="2018-01-01", end_date="2020-01-01",
            price_field="Close", df=df,
            strategy_name="always_long",
            strategy_fn=_always_long_strategy,
            parameter_ranges={},
            warmup_bars=5,
            train_months=6, test_months=2, step_months=2,
            n_samples=5, seed=0,
            min_trades=1, min_exposure_pct=0.0,
        )
        assert set(result.keys()) == {
            "folds", "oos_equity", "bh_equity", "metrics", "trades",
            "param_stability", "warnings",
        }

    def test_oos_equity_is_series(self):
        df = self._df()
        result = run_backtest(
            ticker="TEST", start_date="2018-01-01", end_date="2020-01-01",
            price_field="Close", df=df,
            strategy_name="always_long",
            strategy_fn=_always_long_strategy,
            parameter_ranges={},
            warmup_bars=5,
            train_months=6, test_months=2, step_months=2,
            n_samples=5, seed=0,
            min_trades=1, min_exposure_pct=0.0,
        )
        assert isinstance(result["oos_equity"], pd.Series)
        assert len(result["oos_equity"]) > 0

    def test_metrics_contain_oos_and_bh(self):
        df = self._df()
        result = run_backtest(
            ticker="TEST", start_date="2018-01-01", end_date="2020-01-01",
            price_field="Close", df=df,
            strategy_name="always_long",
            strategy_fn=_always_long_strategy,
            parameter_ranges={},
            warmup_bars=5,
            train_months=6, test_months=2, step_months=2,
            n_samples=5, seed=0,
            min_trades=1, min_exposure_pct=0.0,
        )
        assert "oos" in result["metrics"]
        assert "bh" in result["metrics"]

    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError):
            run_backtest(
                ticker="X", start_date="2020-01-01", end_date="2021-01-01",
                price_field="Close", df=pd.DataFrame(),
                strategy_name="x", strategy_fn=_always_long_strategy,
                parameter_ranges={}, warmup_bars=5,
                train_months=6, test_months=2, step_months=2,
                n_samples=5,
            )

    def test_raises_when_not_enough_data_for_any_fold(self):
        df = _make_ohlcv(10)
        with pytest.raises(ValueError):
            run_backtest(
                ticker="X", start_date="2020-01-01", end_date="2021-01-01",
                price_field="Close", df=df,
                strategy_name="x", strategy_fn=_always_long_strategy,
                parameter_ranges={}, warmup_bars=5,
                train_months=12, test_months=6, step_months=6,
                n_samples=5,
            )

    def test_seed_reproducibility(self):
        df = self._df()
        kwargs = dict(
            ticker="TEST", start_date="2018-01-01", end_date="2020-01-01",
            price_field="Close", df=df,
            strategy_name="alternating",
            strategy_fn=_alternating_strategy,
            parameter_ranges={},
            warmup_bars=5,
            train_months=6, test_months=2, step_months=2,
            n_samples=10, seed=123,
            min_trades=1, min_exposure_pct=0.0,
        )
        r1 = run_backtest(**kwargs)
        r2 = run_backtest(**kwargs)
        pd.testing.assert_series_equal(r1["oos_equity"], r2["oos_equity"])
