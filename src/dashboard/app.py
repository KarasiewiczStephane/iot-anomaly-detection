"""Main Streamlit dashboard for IoT anomaly detection."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import Config

st.set_page_config(page_title="IoT Anomaly Detection", page_icon="🔍", layout="wide")


@st.cache_resource
def load_config() -> Config:
    """Load and cache the application config."""
    return Config.load()


config = load_config()
DB_PATH = config.get("database.path", "data/anomaly_detection.db")


def get_db_connection() -> sqlite3.Connection:
    """Return a connection to the anomaly detection database."""
    return sqlite3.connect(DB_PATH)


# ---- Sidebar filters ----
st.sidebar.title("Filters")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Time"],
)

sensor_types = st.sidebar.multiselect(
    "Sensor Types",
    ["temperature", "vibration", "pressure", "power"],
    default=["temperature", "vibration", "pressure", "power"],
)

TIME_DELTAS = {
    "Last Hour": timedelta(hours=1),
    "Last 6 Hours": timedelta(hours=6),
    "Last 24 Hours": timedelta(hours=24),
    "Last 7 Days": timedelta(days=7),
    "All Time": timedelta(days=365),
}
start_time = datetime.now() - TIME_DELTAS[time_range]

# ---- Main content ----
st.title("IoT Anomaly Detection Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Real-Time View", "Root Cause Analysis", "Sensor Health", "Alert Log", "Model Comparison"]
)

# ----- Tab 1: Real-Time View -----
with tab1:
    st.header("Real-Time Sensor Readings")
    conn = get_db_connection()
    sensor_list = ",".join(f"'{s}'" for s in sensor_types)
    query = (
        f"SELECT timestamp, sensor_type, sensor_id, value, "
        f"is_anomaly_ground_truth as is_anomaly "
        f"FROM sensor_readings "
        f"WHERE timestamp > '{start_time.isoformat()}' "
        f"AND sensor_type IN ({sensor_list}) "
        f"ORDER BY timestamp"
    )
    df = pd.read_sql_query(query, conn)
    conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for sensor_type in sensor_types:
            sensor_df = df[df["sensor_type"] == sensor_type]
            if sensor_df.empty:
                continue
            fig = px.line(
                sensor_df,
                x="timestamp",
                y="value",
                color="sensor_id",
                title=f"{sensor_type.title()} Readings",
            )
            anomalies = sensor_df[sensor_df["is_anomaly"] == 1]
            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies["timestamp"],
                        y=anomalies["value"],
                        mode="markers",
                        marker={"color": "red", "size": 10, "symbol": "x"},
                        name="Anomalies",
                    )
                )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sensor data available for the selected time range.")

# ----- Tab 2: Root Cause Analysis -----
with tab2:
    st.header("Root Cause Analysis")
    conn = get_db_connection()
    anomaly_query = (
        f"SELECT id, timestamp, sensor_id, anomaly_score, contributing_features "
        f"FROM anomaly_log WHERE timestamp > '{start_time.isoformat()}' "
        f"ORDER BY timestamp DESC LIMIT 100"
    )
    anomalies_df = pd.read_sql_query(anomaly_query, conn)
    conn.close()

    if not anomalies_df.empty:
        selected_idx = st.selectbox(
            "Select Anomaly to Analyze",
            range(len(anomalies_df)),
            format_func=lambda i: (
                f"{anomalies_df.iloc[i]['timestamp']} - "
                f"Score: {anomalies_df.iloc[i]['anomaly_score']:.2f}"
            ),
        )
        selected = anomalies_df.iloc[selected_idx]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anomaly Score", f"{selected['anomaly_score']:.2f}")
            st.metric("Sensor", selected["sensor_id"])
        with col2:
            if selected["contributing_features"]:
                contributions = json.loads(selected["contributing_features"])
                contrib_df = pd.DataFrame(
                    [{"Sensor": k, "Contribution": v} for k, v in contributions.items()]
                ).sort_values("Contribution", ascending=False)
                fig = px.bar(
                    contrib_df,
                    x="Sensor",
                    y="Contribution",
                    title="Feature Contributions to Anomaly",
                    color="Contribution",
                    color_continuous_scale="Reds",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies detected in the selected time range.")

# ----- Tab 3: Sensor Health -----
with tab3:
    st.header("Sensor Health Overview")
    conn = get_db_connection()
    health_query = (
        f"SELECT sensor_id, COUNT(*) as total_readings, "
        f"SUM(CASE WHEN is_anomaly_ground_truth = 1 THEN 1 ELSE 0 END) as anomaly_count "
        f"FROM sensor_readings WHERE timestamp > '{start_time.isoformat()}' "
        f"GROUP BY sensor_id"
    )
    health_df = pd.read_sql_query(health_query, conn)
    conn.close()

    if not health_df.empty:
        health_df["anomaly_rate"] = health_df["anomaly_count"] / health_df["total_readings"]
        health_df["health_status"] = health_df["anomaly_rate"].apply(
            lambda x: "Healthy" if x < 0.05 else "Warning" if x < 0.1 else "Critical"
        )
        cols = st.columns(min(4, len(health_df)))
        for i, row in health_df.iterrows():
            with cols[int(i) % len(cols)]:
                st.metric(
                    row["sensor_id"],
                    row["health_status"],
                    f"{row['anomaly_rate'] * 100:.1f}% anomaly rate",
                )
        fig = px.bar(
            health_df,
            x="sensor_id",
            y="anomaly_rate",
            title="Anomaly Rate by Sensor",
            color="anomaly_rate",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sensor health data available.")

# ----- Tab 4: Alert Log -----
with tab4:
    st.header("Alert History")
    conn = get_db_connection()
    alerts_query = (
        f"SELECT timestamp, alert_type, severity, message, resolved "
        f"FROM alert_history WHERE timestamp > '{start_time.isoformat()}' "
        f"ORDER BY timestamp DESC"
    )
    alerts_df = pd.read_sql_query(alerts_query, conn)
    conn.close()

    if not alerts_df.empty:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["low", "medium", "high", "critical"],
            default=["high", "critical"],
        )
        filtered_alerts = alerts_df[alerts_df["severity"].isin(severity_filter)]
        st.dataframe(filtered_alerts, use_container_width=True)
        csv = filtered_alerts.to_csv(index=False)
        st.download_button("Export Alerts CSV", csv, "alerts_export.csv", "text/csv")
    else:
        st.info("No alerts in the selected time range.")

# ----- Tab 5: Model Comparison -----
with tab5:
    st.header("Model Performance Comparison")
    from pathlib import Path

    eval_path = Path("data/evaluation_results.json")
    if eval_path.exists():
        with open(eval_path) as fh:
            eval_results = json.load(fh)

        metrics_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Precision": results["precision"],
                    "Recall": results["recall"],
                    "F1 Score": results["f1"],
                    "AUC-ROC": results["auc_roc"],
                }
                for name, results in eval_results.items()
            ]
        )

        st.subheader("Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        # Radar chart
        fig = go.Figure()
        for name, results in eval_results.items():
            fig.add_trace(
                go.Scatterpolar(
                    r=[results["precision"], results["recall"], results["f1"], results["auc_roc"]],
                    theta=["Precision", "Recall", "F1", "AUC-ROC"],
                    fill="toself",
                    name=name,
                )
            )
        fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}})
        st.plotly_chart(fig, use_container_width=True)

        # Detection results by method
        st.subheader("Detection Results by Method")
        conn = get_db_connection()
        method_query = (
            "SELECT timestamp, detection_method, anomaly_score, sensor_id "
            "FROM anomaly_log WHERE timestamp > ? ORDER BY timestamp"
        )
        method_df = pd.read_sql_query(method_query, conn, params=[start_time.isoformat()])
        conn.close()

        if not method_df.empty:
            method_df["timestamp"] = pd.to_datetime(method_df["timestamp"])
            fig = px.scatter(
                method_df,
                x="timestamp",
                y="anomaly_score",
                color="detection_method",
                title="Anomaly Scores by Detection Method",
                hover_data=["sensor_id"],
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run model training and evaluation to see comparison results.")

# ---- Sidebar export ----
st.sidebar.markdown("---")
st.sidebar.subheader("Export Options")

if st.sidebar.button("Generate Full Report"):
    report_data: dict = {
        "generated_at": datetime.now().isoformat(),
        "time_range": time_range,
        "sensor_types": sensor_types,
        "summary": {},
    }
    conn = get_db_connection()
    summary_query = (
        "SELECT COUNT(*) as total_readings, "
        "SUM(CASE WHEN is_anomaly_ground_truth = 1 THEN 1 ELSE 0 END) as total_anomalies, "
        "COUNT(DISTINCT sensor_id) as sensor_count "
        "FROM sensor_readings WHERE timestamp > ?"
    )
    summary = pd.read_sql_query(summary_query, conn, params=[start_time.isoformat()])
    conn.close()
    report_data["summary"] = summary.iloc[0].to_dict()

    st.sidebar.download_button(
        "Download Report (JSON)",
        json.dumps(report_data, indent=2, default=str),
        f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json",
    )

# ---- Footer ----
st.sidebar.markdown("---")
st.sidebar.markdown("**IoT Anomaly Detection v1.0**")
