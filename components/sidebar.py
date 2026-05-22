import streamlit as st
from datetime import date, timedelta
from pathlib import Path
from strategies import discover_strategies, STRATEGIES_DIR
from components.strategy_card import render_strategy_card

_LLM_PROMPT_TEMPLATE = '''\
STRATEGY_INFO = {
    "name": "My Strategy",
    "description": "...",
    "parameters": {"param1": default_value},
    "parameter_ranges": {"param1": {"min": 1, "max": 10, "step": 1}},
    "warmup_bars": 50,
    "tags": ["tag1", "tag2"],
}

def generate_signals(df, param1):
    # df has columns: Open, High, Low, Close (filtered), Volume
    # Return pd.Series of int (1=long, 0=flat, -1=short), same index as df
    ...
'''


def render_sidebar(strategies: dict) -> dict:
    """Render sidebar controls and return config dict."""
    today = date.today()
    three_years_ago = today - timedelta(days=3 * 365)

    # --- Data ---
    with st.expander("Data", expanded=False):
        ticker = st.text_input("Ticker", value="SPY").strip().upper()
        start_date = st.date_input("Start Date", value=three_years_ago)
        end_date = st.date_input("End Date", value=today)
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        price_field = st.selectbox("Price Field", ["Close", "Adj Close"], index=0)

    # --- Capital & Sizing ---
    with st.expander("Capital & Sizing", expanded=False):
        initial_capital = st.number_input(
            "Initial Capital ($)", value=10000, step=1000, min_value=0
        )
        sizing_mode_label = st.radio(
            "Sizing Mode", ["% of Equity", "Fixed Dollar"], index=0
        )
        sizing_mode = "pct_equity" if sizing_mode_label == "% of Equity" else "fixed_dollar"

        if sizing_mode == "pct_equity":
            sizing_value = st.number_input(
                "Position Size (fraction of equity)",
                value=1.0,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
            )
        else:
            sizing_value = st.number_input(
                "Position Size ($)", value=1000.0, min_value=0.0, step=100.0
            )

    # --- Costs ---
    with st.expander("Costs", expanded=False):
        fee_pct_input = st.number_input(
            "Fee (%)", value=0.1, min_value=0.0, step=0.01, format="%.3f"
        )
        slippage_pct_input = st.number_input(
            "Slippage (%)", value=0.1, min_value=0.0, step=0.01, format="%.3f"
        )
        fee_pct = fee_pct_input / 100.0
        slippage_pct = slippage_pct_input / 100.0

    # --- Walk-Forward ---
    with st.expander("Walk-Forward", expanded=False):
        train_months = st.slider("Train Months", min_value=3, max_value=60, value=12)
        test_months = st.slider("Test Months", min_value=1, max_value=24, value=3)
        step_months = st.slider("Step Months", min_value=1, max_value=24, value=3)

        if step_months < test_months:
            st.warning("⚠️ step_months < test_months: OOS windows overlap.")

        n_samples = st.slider("# Samples (optimization)", min_value=10, max_value=1000, value=100)
        seed = st.number_input("Random Seed", value=42, step=1)
        min_trades = st.number_input("Min Trades", value=10, step=1, min_value=0)
        min_exposure_pct = st.number_input(
            "Min Exposure (%)", value=5.0, step=0.5, min_value=0.0, format="%.1f"
        )

    # --- Strategy ---
    with st.expander("Strategy", expanded=True):
        strategy_names = list(strategies.keys())

        if strategy_names:
            strategy_name = st.selectbox("Strategy", strategy_names)
            render_strategy_card(strategies[strategy_name]["info"])
        else:
            strategy_name = None
            st.info("No strategies loaded. Upload a strategy file below.")

        uploaded_file = st.file_uploader(
            "Upload Strategy (.py)", type=["py"], label_visibility="visible"
        )
        if uploaded_file is not None:
            dest = STRATEGIES_DIR / uploaded_file.name
            dest.write_bytes(uploaded_file.getvalue())
            st.session_state["strategies"] = discover_strategies(STRATEGIES_DIR)
            st.rerun()

        st.markdown("**Strategy template:**")
        st.code(_LLM_PROMPT_TEMPLATE, language="python")

    # --- Run button ---
    run_clicked = st.button("▶ Run Backtest", use_container_width=True)

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "price_field": price_field,
        "initial_capital": initial_capital,
        "sizing_mode": sizing_mode,
        "sizing_value": sizing_value,
        "fee_pct": fee_pct,
        "slippage_pct": slippage_pct,
        "train_months": train_months,
        "test_months": test_months,
        "step_months": step_months,
        "n_samples": n_samples,
        "seed": seed,
        "min_trades": min_trades,
        "min_exposure_pct": min_exposure_pct,
        "strategy_name": strategy_name,
        "run_clicked": run_clicked,
    }
