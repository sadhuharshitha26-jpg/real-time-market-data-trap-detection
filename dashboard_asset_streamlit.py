"""
Asset-aware, near-real-time MarketTrap dashboard.
Feels like TradingView, not a research paper.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from asset_registry import supported_symbols, get_asset_params
from feature_engineering.asset_trap_features import compute_asset_trap_features
from ml_pipeline.anomaly_model import IsolationForestModel
from risk_inference.asset_trap_risk import compute_asset_trap_risk
from realtime_feed import start_ws, get_latest_ohlcv, get_connection_status, tick_buffers

st.set_page_config(page_title="MarketTrap", layout="wide")

st.title("MarketTrap – Retail Trap Risk Explainer")

# Sidebar: asset selection and refresh
with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Select Cryptocurrency", supported_symbols())
    api_key = st.text_input("CryptoCompare API Key", type="password", value="beca920776b27dd34f4e790646bc3e7d16cc5abe9e10e77e2ffcc7e000823b79")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 3, 15, 5)

    if st.button("Start WebSocket Feed") and api_key:
        if "ws_thread" not in st.session_state or not st.session_state.ws_thread.is_alive():
            import threading
            st.session_state.ws_thread = threading.Thread(
                target=start_ws, args=(api_key,), daemon=True
            )
            st.session_state.ws_thread.start()
            st.success("WebSocket started.")
        else:
            st.info("WebSocket already running.")

    st.markdown("---")
    st.subheader("Connection Status")
    status = get_connection_status()
    if status["connected"]:
        st.success(f"WebSocket: Connected ({status['connection_age_seconds']:.0f}s ago)")
    else:
        st.error("WebSocket: Not connected")

    st.write("Symbol tick freshness:")
    for sym, info in status["symbols"].items():
        freshness = "✅ Fresh" if info["fresh"] else "⚠️ Stale"
        st.write(f"- {sym}: {info['buffer_size']} ticks, last {info['last_tick_age_seconds']:.0f}s ago {freshness}")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"{symbol} – Live Chart")
    chart_placeholder = st.empty()

with col2:
    st.subheader("Risk Summary")
    risk_placeholder = st.empty()
    reasons_placeholder = st.empty()
    invalidation_placeholder = st.empty()

def run_pipeline(symbol: str):
    """Fetch live data, compute features, and infer trap risk."""
    df = get_latest_ohlcv(symbol, min_ticks=5, window_seconds=60)  # lowered to 5 for debugging
    if df is None or df.empty:
        return None, None, "Not enough ticks to build OHLCV"
    features = compute_asset_trap_features(df, symbol)
    # Load anomaly model once per session
    if "anomaly_model" not in st.session_state:
        st.session_state.anomaly_model = IsolationForestModel.load("models/isolation_forest.pkl")
    anomaly_scores = st.session_state.anomaly_model.anomaly_score(features)
    result = compute_asset_trap_risk(features, symbol, anomaly_score=float(anomaly_scores[-1]))
    return df, result, None

def render_chart(df: pd.DataFrame, result):
    """Render price + risk chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"{symbol} Price", "Trap Risk %"),
        row_heights=[0.65, 0.35]
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["close"], name="Price", line=dict(color="#1f77b4")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=[result.trap_risk_score] * len(df),
            name="Current Risk",
            line=dict(color="#ff7f0e", dash="dash")
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Risk %", range=[0, 100], row=2, col=1)
    fig.update_layout(height=600, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
    return fig

def render_risk_summary(result):
    """Render risk score, level, and reasons."""
    risk_placeholder.metric(
        label="Current Trap Risk",
        value=f"{result.trap_risk_score:.1f}%",
        delta=result.risk_level
    )
    if result.top_3_reasons:
        reasons_placeholder.subheader("Why this looks risky")
        for reason in result.top_3_reasons:
            reasons_placeholder.write("• " + reason)
    if result.invalidated_by:
        invalidation_placeholder.subheader("What would invalidate this trap")
        for inv in result.invalidated_by:
            invalidation_placeholder.write("• " + inv)

# Initial load
df, result, msg = run_pipeline(symbol)
if df is not None and result is not None:
    chart_placeholder.plotly_chart(render_chart(df, result), use_container_width=True)
    render_risk_summary(result)
else:
    st.warning(f"Waiting for live data… {msg or 'Unknown error'}")

# Auto-refresh loop
if auto_refresh:
    while True:
        time.sleep(refresh_interval)
        df, result, msg = run_pipeline(symbol)
        if df is not None and result is not None:
            chart_placeholder.plotly_chart(render_chart(df, result), use_container_width=True)
            render_risk_summary(result)
        else:
            st.warning(f"Waiting for live data… {msg or 'Unknown error'}")
