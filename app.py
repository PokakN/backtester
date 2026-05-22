import streamlit as st
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests
from strategies import STRATEGIES
from backtester import run_backtest
from components.sidebar import render_sidebar
from components.charts import (
    build_gantt_chart, build_equity_chart, build_drawdown_chart,
    build_monthly_heatmap, build_param_stability_table,
)

st.set_page_config(layout="wide", page_title="Walk-Forward Backtester")

# Strategy registry (cached in session_state)
if "strategies" not in st.session_state:
    st.session_state["strategies"] = STRATEGIES

with st.sidebar:
    config = render_sidebar(st.session_state["strategies"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, start: str, end: str, price_field: str) -> pd.DataFrame:
    # impersonate bypasses Yahoo Finance bot detection; verify=False skips Windows SSL cert issues
    session = curl_requests.Session(impersonate="chrome136", verify=False)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, session=session)
    if df.empty:
        return df
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Normalize timezone (tz_convert(None) strips tz on tz-aware index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    # Use price_field as "Close"
    if price_field == "Adj Close" and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df


# Backtest execution — gate ALL execution on config["run_clicked"]
if config["run_clicked"]:
    strategy_name = config["strategy_name"]
    if not strategy_name or strategy_name not in st.session_state["strategies"]:
        st.error("Please select a valid strategy.")
        st.stop()

    with st.spinner("Fetching data..."):
        df = fetch_data(
            config["ticker"],
            str(config["start_date"]),
            str(config["end_date"]),
            config["price_field"],
        )

    if df.empty:
        st.error(f"No data returned for '{config['ticker']}'. Check ticker and date range.")
        st.stop()

    min_bars_needed = (config["train_months"] + config["test_months"]) * 21
    if len(df) < min_bars_needed:
        st.error(f"Not enough data: {len(df)} bars, need ~{min_bars_needed}.")
        st.stop()

    strategy = st.session_state["strategies"][strategy_name]
    progress_bar = st.progress(0)

    try:
        with st.spinner("Running walk-forward backtest..."):
            result = run_backtest(
                ticker=config["ticker"],
                start_date=str(config["start_date"]),
                end_date=str(config["end_date"]),
                price_field=config["price_field"],
                df=df,
                strategy_name=strategy_name,
                strategy_fn=strategy["fn"],
                parameter_ranges=strategy["info"]["parameter_ranges"],
                warmup_bars=strategy["info"]["warmup_bars"],
                train_months=config["train_months"],
                test_months=config["test_months"],
                step_months=config["step_months"],
                n_samples=config["n_samples"],
                seed=config["seed"],
                min_trades=config["min_trades"],
                min_exposure_pct=config["min_exposure_pct"],
                fee_pct=config["fee_pct"],
                slippage_pct=config["slippage_pct"],
                sizing_mode=config["sizing_mode"],
                sizing_value=config["sizing_value"],
                initial_capital=config["initial_capital"],
                progress_callback=lambda i, n, _: progress_bar.progress(i / max(n, 1)),
            )
        st.session_state["result"] = result
        st.session_state["result_initial_capital"] = config["initial_capital"]
    except Exception as e:
        st.error(f"Backtest failed: {e}")
    finally:
        progress_bar.empty()


# Dashboard display — show if result exists in session_state
if "result" in st.session_state:
    result = st.session_state["result"]
    folds = result["folds"]
    oos_equity = result["oos_equity"]
    bh_equity = result["bh_equity"]
    metrics = result["metrics"]
    trades = result["trades"]
    param_stability = result["param_stability"]

    # Warnings
    for w in result.get("warnings", []):
        st.warning(w)
    skipped = [f for f in folds if f.get("skipped")]
    if skipped:
        st.warning(f"{len(skipped)} fold(s) were skipped (all samples rejected).")

    initial_capital = st.session_state.get("result_initial_capital", 10_000)

    # Section 1: Fold Timeline (Gantt)
    st.subheader("Fold Timeline")
    st.plotly_chart(build_gantt_chart(folds), use_container_width=True)

    # Section 2: Summary Metrics + P&L Cards
    st.subheader("Performance Summary")

    oos_m = metrics["oos"]
    bh_m = metrics["bh"]
    metrics_table = pd.DataFrame({
        "OOS Strategy": [
            f"{oos_m['total_return_pct']:.2f}%",
            f"{oos_m['annual_return_pct']:.2f}%",
            f"{oos_m['max_drawdown_pct']:.2f}%",
            f"{oos_m['sharpe']:.3f}",
            f"{oos_m['win_rate_pct']:.1f}%" if not pd.isna(oos_m.get('win_rate_pct', float('nan'))) else "N/A",
            f"{oos_m['avg_trade_pct']:.2f}%" if not pd.isna(oos_m.get('avg_trade_pct', float('nan'))) else "N/A",
        ],
        "Buy & Hold": [
            f"{bh_m['total_return_pct']:.2f}%",
            f"{bh_m['annual_return_pct']:.2f}%",
            f"{bh_m['max_drawdown_pct']:.2f}%",
            f"{bh_m['sharpe']:.3f}",
            "N/A",
            "N/A",
        ],
    }, index=["Total Return", "Annual Return", "Max Drawdown", "Sharpe", "Win Rate", "Avg Trade"])
    st.dataframe(metrics_table, use_container_width=True)

    # 4 P&L summary cards
    col1, col2, col3, col4 = st.columns(4)
    net_pnl = oos_m["final_equity"] - initial_capital
    bh_pnl = bh_m["final_equity"] - initial_capital
    with col1:
        st.metric("Net P&L", f"${net_pnl:,.0f}", delta=f"vs B&H: ${net_pnl - bh_pnl:+,.0f}")
    with col2:
        st.metric("Annual Return", f"{oos_m['annual_return_pct']:.2f}%",
                  delta=f"{oos_m['annual_return_pct'] - bh_m['annual_return_pct']:+.2f}% vs B&H")
    with col3:
        st.metric("Max Drawdown", f"{oos_m['max_drawdown_pct']:.2f}%",
                  delta=f"{oos_m['max_drawdown_pct'] - bh_m['max_drawdown_pct']:+.2f}% vs B&H",
                  delta_color="inverse")
    with col4:
        st.metric("Sharpe", f"{oos_m['sharpe']:.3f}",
                  delta=f"{oos_m['sharpe'] - bh_m['sharpe']:+.3f} vs B&H")

    # Section 3: Equity Curves
    st.subheader("Equity Curves")
    st.plotly_chart(build_equity_chart(oos_equity, bh_equity, initial_capital), use_container_width=True)

    # Section 4: Drawdown
    st.subheader("Drawdown")
    st.plotly_chart(build_drawdown_chart(oos_equity, bh_equity), use_container_width=True)

    # Section 5: Monthly Heatmap
    st.subheader("Monthly Returns Heatmap")
    st.plotly_chart(build_monthly_heatmap(oos_equity), use_container_width=True)

    # Section 6: Parameter Stability
    st.subheader("Parameter Stability")
    if param_stability:
        param_names = list(param_stability.keys())
        stab_df = build_param_stability_table(folds, param_names)
        if not stab_df.empty:
            st.dataframe(stab_df, use_container_width=True)

        # Stability verdict
        all_stable = all(param_stability[p]["stable"] for p in param_names)
        if all_stable:
            st.success("✅ Stable — parameters consistent across folds.")
        else:
            st.warning("⚠️ Unstable — parameters vary significantly. May not be robust for live trading.")

        # Suggested params code block
        medians = {p: param_stability[p]["median"] for p in param_names}
        suggested = "\n".join(f'    "{k}": {int(v) if float(v) == int(v) else v},' for k, v in medians.items())
        st.code(f"best_params = {{\n{suggested}\n}}", language="python")
    else:
        st.info("No parameter stability data available.")

    # Section 7: Trade Log
    st.subheader("Trade Log")
    if not trades.empty:
        st.dataframe(trades, use_container_width=True)
    else:
        st.info("No trades recorded.")
