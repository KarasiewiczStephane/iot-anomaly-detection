"""Reusable Plotly chart components for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_anomaly_timeline(
    df: pd.DataFrame,
    sensor_col: str = "sensor_id",
    value_col: str = "value",
    time_col: str = "timestamp",
    anomaly_col: str = "is_anomaly",
) -> go.Figure:
    """Create a timeline chart with anomaly markers overlaid.

    Args:
        df: DataFrame with sensor readings.
        sensor_col: Column identifying the sensor.
        value_col: Column with measured values.
        time_col: Column with timestamps.
        anomaly_col: Binary column flagging anomalies.

    Returns:
        Plotly figure.
    """
    fig = px.line(df, x=time_col, y=value_col, color=sensor_col)

    anomalies = df[df[anomaly_col] == 1]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies[time_col],
                y=anomalies[value_col],
                mode="markers",
                marker={"color": "red", "size": 10, "symbol": "x"},
                name="Anomalies",
                showlegend=True,
            )
        )

    return fig


def create_contribution_chart(
    contributions: dict[str, float],
    title: str = "Feature Contributions",
) -> go.Figure:
    """Create a bar chart for feature contribution scores.

    Args:
        contributions: Mapping of feature name to contribution score.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    sorted_items = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    fig = go.Figure(
        go.Bar(
            x=[item[0] for item in sorted_items],
            y=[item[1] for item in sorted_items],
            marker_color=[
                px.colors.sequential.Reds[min(int(item[1] * 9), 8)] for item in sorted_items
            ],
        )
    )
    fig.update_layout(title=title, xaxis_title="Sensor", yaxis_title="Contribution Score")
    return fig


def create_health_gauge(anomaly_rate: float, sensor_name: str) -> go.Figure:
    """Create a gauge chart for sensor health status.

    Args:
        anomaly_rate: Anomaly rate in [0, 1].
        sensor_name: Sensor identifier used as chart title.

    Returns:
        Plotly gauge figure.
    """
    color = "green" if anomaly_rate < 0.05 else "yellow" if anomaly_rate < 0.1 else "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=anomaly_rate * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": sensor_name},
            gauge={
                "axis": {"range": [0, 20]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 5], "color": "lightgreen"},
                    {"range": [5, 10], "color": "lightyellow"},
                    {"range": [10, 20], "color": "lightcoral"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 10,
                },
            },
        )
    )

    return fig
