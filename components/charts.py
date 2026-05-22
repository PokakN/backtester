import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_gantt_chart(folds: list[dict]) -> go.Figure:
    """Horizontal bar chart showing train (blue) and test (orange) windows per fold."""
    fig = go.Figure()

    fold_labels = []
    train_bases = []
    train_durations = []
    test_bases = []
    test_durations = []

    for i, fold in enumerate(folds):
        train_start = fold.get("train_start")
        test_start = fold.get("test_start")

        if train_start is None or test_start is None:
            continue

        label = f"Fold {i + 1}"
        fold_labels.append(label)

        train_end = fold.get("train_end", test_start)
        test_end = fold.get("test_end", test_start)

        train_base = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        test_base = pd.Timestamp(test_start)
        test_end_ts = pd.Timestamp(test_end)

        # Plotly date-axis bars: base = ms since epoch, x = duration in ms
        ms_per_day = 24 * 3600 * 1000
        train_bases.append(train_base.value // 10**6)
        train_durations.append((train_end_ts - train_base).days * ms_per_day)
        test_bases.append(test_base.value // 10**6)
        test_durations.append((test_end_ts - test_base).days * ms_per_day)

    fig.add_trace(
        go.Bar(
            name="Train",
            y=fold_labels,
            x=train_durations,
            base=train_bases,
            orientation="h",
            marker_color="steelblue",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Test",
            y=fold_labels,
            x=test_durations,
            base=test_bases,
            orientation="h",
            marker_color="darkorange",
        )
    )

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            type="date",
            title="Date",
        ),
        yaxis=dict(title="Fold"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=20, t=40, b=40),
    )

    return fig


def build_equity_chart(
    oos_equity: pd.Series, bh_equity: pd.Series, initial_capital: float
) -> go.Figure:
    """Line chart comparing OOS equity vs Buy & Hold equity."""
    fig = go.Figure()

    oos_custom = ((oos_equity / initial_capital - 1) * 100).values
    bh_custom = ((bh_equity / initial_capital - 1) * 100).values

    fig.add_trace(
        go.Scatter(
            x=oos_equity.index,
            y=oos_equity.values,
            customdata=oos_custom,
            name="OOS Strategy",
            line=dict(color="green"),
            hovertemplate="Date: %{x}<br>Value: %{y:$,.0f}<br>Return: %{customdata:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bh_equity.index,
            y=bh_equity.values,
            customdata=bh_custom,
            name="Buy & Hold",
            line=dict(color="grey"),
            hovertemplate="Date: %{x}<br>Value: %{y:$,.0f}<br>Return: %{customdata:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            title="Date",
        ),
        yaxis=dict(
            title="Portfolio Value ($)",
            tickformat="$,.0f",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=20, t=40, b=40),
    )

    return fig


def build_drawdown_chart(oos_equity: pd.Series, bh_equity: pd.Series) -> go.Figure:
    """Area chart showing rolling drawdown % for OOS and B&H."""
    oos_max = oos_equity.cummax().replace(0, np.nan)
    bh_max = bh_equity.cummax().replace(0, np.nan)
    oos_dd = ((oos_equity - oos_max) / oos_max * 100).fillna(0)
    bh_dd = ((bh_equity - bh_max) / bh_max * 100).fillna(0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=oos_dd.index,
            y=oos_dd.values,
            name="OOS Strategy",
            fill="tozeroy",
            line=dict(color="green"),
            fillcolor="rgba(0, 128, 0, 0.2)",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bh_dd.index,
            y=bh_dd.values,
            name="Buy & Hold",
            fill="tozeroy",
            line=dict(color="grey"),
            fillcolor="rgba(128, 128, 128, 0.2)",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Drawdown (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=20, t=40, b=40),
    )

    return fig


def build_monthly_heatmap(oos_equity: pd.Series) -> go.Figure:
    """Heatmap of monthly returns (years x months)."""
    monthly = oos_equity.resample("ME").last().pct_change() * 100
    monthly.index = monthly.index.to_period("M")

    years = sorted(monthly.index.year.unique())
    if not len(years):
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for monthly heatmap",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    months_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    matrix = [
        [monthly.get(pd.Period(f"{y}-{m:02d}", "M"), np.nan) for m in range(1, 13)]
        for y in years
    ]

    text_matrix = [
        [f"{v:.1f}" if not np.isnan(v) else "" for v in row]
        for row in matrix
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=months_labels,
            y=[str(y) for y in years],
            text=text_matrix,
            texttemplate="%{text}%",
            colorscale=[[0, "red"], [0.5, "lightgrey"], [1, "green"]],
            zmin=-10,
            zmax=10,
            colorbar=dict(title="Return (%)"),
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(title="Month"),
        yaxis=dict(title="Year", autorange="reversed"),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return fig


def build_param_stability_table(folds: list[dict], param_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame with param values per fold (columns=fold numbers, rows=params)."""
    data = {}

    for i, fold in enumerate(folds):
        if fold.get("skipped", False):
            continue
        best_params = fold.get("best_params")
        if best_params is None:
            continue

        fold_label = f"Fold {i + 1}"
        data[fold_label] = {p: best_params.get(p) for p in param_names}

    if not data:
        return pd.DataFrame(index=param_names)

    df = pd.DataFrame(data, index=param_names)
    return df
