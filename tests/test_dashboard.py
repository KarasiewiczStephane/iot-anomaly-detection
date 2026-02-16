"""Tests for dashboard reusable chart components."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.dashboard.components import (
    create_anomaly_timeline,
    create_contribution_chart,
    create_health_gauge,
)


@pytest.fixture()
def sensor_df() -> pd.DataFrame:
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(50):
        rows.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "sensor_id": "temp_00",
                "value": 70.0 + i * 0.1,
                "is_anomaly": 1 if i in (10, 20, 30) else 0,
            }
        )
    return pd.DataFrame(rows)


class TestCreateAnomalyTimeline:
    def test_returns_figure(self, sensor_df: pd.DataFrame) -> None:
        fig = create_anomaly_timeline(sensor_df)
        assert isinstance(fig, go.Figure)

    def test_has_anomaly_trace(self, sensor_df: pd.DataFrame) -> None:
        fig = create_anomaly_timeline(sensor_df)
        trace_names = [t.name for t in fig.data]
        assert "Anomalies" in trace_names

    def test_no_anomaly_trace_when_none(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "sensor_id": ["s1"],
                "value": [1.0],
                "is_anomaly": [0],
            }
        )
        fig = create_anomaly_timeline(df)
        trace_names = [t.name for t in fig.data]
        assert "Anomalies" not in trace_names


class TestCreateContributionChart:
    def test_returns_figure(self) -> None:
        fig = create_contribution_chart({"temp": 0.9, "vib": 0.3})
        assert isinstance(fig, go.Figure)

    def test_bar_count(self) -> None:
        fig = create_contribution_chart({"a": 0.5, "b": 0.2, "c": 0.8})
        bar_trace = fig.data[0]
        assert len(bar_trace.x) == 3


class TestCreateHealthGauge:
    def test_returns_figure(self) -> None:
        fig = create_health_gauge(0.03, "temp_00")
        assert isinstance(fig, go.Figure)

    def test_gauge_value(self) -> None:
        fig = create_health_gauge(0.05, "sensor")
        indicator = fig.data[0]
        assert indicator.value == pytest.approx(5.0)
