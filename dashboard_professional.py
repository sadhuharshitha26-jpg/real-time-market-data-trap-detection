"""
MarketTrap Professional - Institutional-Grade Crypto Intelligence Terminal
Professional real-time dashboard with Binance WebSocket integration and advanced risk detection.
"""

from collections import deque
from datetime import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.binance_ws import BinanceWSClient
from ml_pipeline.anomaly_model import IsolationForestModel
from risk_inference.realtime_trap_engine import (
    build_component_scores,
    buyer_seller_control,
    classify_trap_type,
    extract_trap_reasons,
)

# --- Configuration & Styling ---
st.set_page_config(
    page_title="MarketTrap Pro Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .terminal-header.alert {
        border-bottom: 1px solid rgba(248, 81, 73, 0.6);
        box-shadow: 0 0 18px rgba(248, 81, 73, 0.35);
    }
    .status-dot.alert {
        background-color: var(--bearish);
        box-shadow: 0 0 12px var(--bearish);
    }
    :root {
        --bg-color: #05070a;
        --panel-bg: #0d1117;
        --border-color: #30363d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --accent-blue: #58a6ff;
        --bullish: #23d18b;
        --bearish: #f85149;
        --warning: #d29922;
    }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1.5rem;
        background: var(--panel-bg);
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }
    .terminal-title {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: -0.5px;
        color: var(--accent-blue);
    }
    .terminal-status {
        font-size: 0.8rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--bullish);
        box-shadow: 0 0 10px var(--bullish);
        animation: pulse 2s infinite;
    }

    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(13, 17, 23, 0.7);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(48, 54, 61, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.15);
    }
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .risk-gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(13, 17, 23, 0.8);
        backdrop-filter: blur(15px);
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(48, 54, 61, 0.5);
        height: 100%;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .trap-type-badge {
        margin-top: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 999px;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        background: #11161d;
        padding: 0.35rem 0.8rem;
        text-align: center;
    }

    .control-chip {
        margin-top: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        text-align: center;
        width: 100%;
    }

    .reasons-panel {
        width: 100%;
        margin-top: 0.9rem;
        background: #0b1117;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.8rem;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }

    [data-testid="stSidebar"] {
        background-color: var(--panel-bg);
        border-right: 1px solid var(--border-color);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

CRITICAL_THRESHOLD = 70.0
CRITICAL_STREAK_UPDATES = 3
MAX_TRAP_EVENTS = 10
EMA_ALPHA = 0.35


# --- State Management ---
if "binance_client" not in st.session_state:
    st.session_state.binance_client = BinanceWSClient(symbols=["btcusdt", "ethusdt", "solusdt", "bnbusdt"])
    st.session_state.binance_client.start()

if "model" not in st.session_state:
    try:
        st.session_state.model = IsolationForestModel.load("models/isolation_forest.pkl")
    except Exception:
        st.session_state.model = None

if "risk_state" not in st.session_state:
    st.session_state.risk_state = {}

if "trap_history" not in st.session_state:
    st.session_state.trap_history = deque(maxlen=MAX_TRAP_EVENTS)

client = st.session_state.binance_client
model = st.session_state.model


# --- UI Components ---
def render_header(is_critical: bool = False):
    now = datetime.utcnow().strftime("%H:%M:%S UTC")
    header_class = "terminal-header alert" if is_critical else "terminal-header"
    dot_class = "status-dot alert" if is_critical else "status-dot"
    status_text = "CRITICAL RISK" if is_critical else "LIVE FEED"

    st.markdown(
        f"""
    <div class="{header_class}">
        <div class="terminal-title">MARKET TRAP DETECTION</div>
        <div class="terminal-status">
            <span>{status_text}</span>
            <div class="{dot_class}"></div>
            <span style="margin-left: 10px; font-family: 'JetBrains Mono'">{now}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metrics(symbol: str):
    df = client.get_latest_data(symbol)
    if df is not None and not df.empty:
        curr = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else curr

        change = curr["price"] - prev["price"]
        change_pct = (change / prev["price"] * 100) if prev["price"] != 0 else 0
        color = "var(--bullish)" if change >= 0 else "var(--bearish)"

        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Price ({symbol.upper()})</div>
                <div class="metric-value">${curr['price']:,.2f}</div>
                <div class="metric-delta" style="color: {color}">
                    {"+" if change >= 0 else ""}{change:.2f} ({change_pct:+.2f}%)
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">24h High</div>
                <div class="metric-value">${curr['high']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">24h Low</div>
                <div class="metric-value">${curr['low']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value">{curr['volume']:,.2f}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return curr
    return None


@st.cache_data(ttl=60)
def fetch_historical_context(symbol: str, limit: int = 120):
    """Fetch recent historical 1-minute data from Binance REST API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval=1m&limit={limit}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )
            df["timestamp"] = df["timestamp"] / 1000.0
            df["price"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            return df[["timestamp", "price", "volume"]]
    except Exception:
        pass
    return pd.DataFrame()


def build_merged_1m_frame(symbol: str):
    rt_df = client.get_latest_data(symbol)
    hist_df = fetch_historical_context(symbol, limit=120)
    frames = []

    if hist_df is not None and not hist_df.empty:
        frames.append(hist_df[["timestamp", "price", "volume"]].copy())

    if rt_df is not None and not rt_df.empty:
        rt_df = rt_df.copy()
        rt_df["datetime"] = pd.to_datetime(rt_df["timestamp"], unit="s")
        rt_df["volume_delta"] = rt_df["volume"].diff().fillna(0).clip(lower=0)
        rt_1m = (
            rt_df.set_index("datetime")
            .resample("1min")
            .agg({"price": "last", "volume_delta": "sum"})
            .dropna()
            .reset_index()
        )
        if not rt_1m.empty:
            rt_1m["timestamp"] = rt_1m["datetime"].astype("int64") // 10**9
            rt_1m = rt_1m.rename(columns={"volume_delta": "volume"})
            frames.append(rt_1m[["timestamp", "price", "volume"]])

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = (
        merged.groupby("timestamp", as_index=False)
        .agg({"price": "last", "volume": "sum"})
        .sort_values("timestamp")
        .tail(180)
    )
    merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="s")
    return merged


def create_advanced_chart(df_1m: pd.DataFrame):
    if df_1m is None or df_1m.empty:
        st.info("Awaiting more ticks for visualization...")
        return

    frame = df_1m.copy()
    frame["datetime"] = frame["datetime"] + pd.to_timedelta(frame.index, unit="ms")
    frame["volume_plot"] = frame["volume"].clip(upper=frame["volume"].quantile(0.95))

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=frame["datetime"],
            y=frame["price"],
            mode="lines+markers",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=4),
            name="Price",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=frame["datetime"],
            y=frame["volume_plot"],
            marker=dict(color="rgba(180,180,180,0.5)"),
            width=1000 * 60,
            name="Volume",
        ),
        secondary_y=False,
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        showlegend=False,
        hovermode="closest",
    )
    fig.update_yaxes(title_text="Volume", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Price", secondary_y=True, showgrid=True, gridcolor="#161b22")

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_risk_gauge(risk_score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "TRAP PROBABILITY", "font": {"size": 14, "color": "#8b949e", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#30363d"},
                "bar": {"color": "#58a6ff"},
                "bgcolor": "#0d1117",
                "borderwidth": 2,
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 30], "color": "rgba(35, 209, 139, 0.1)"},
                    {"range": [30, 70], "color": "rgba(210, 153, 34, 0.1)"},
                    {"range": [70, 100], "color": "rgba(248, 81, 73, 0.1)"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "JetBrains Mono"},
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def compute_realtime_snapshot(symbol: str, df_1m: pd.DataFrame):
    default_snapshot = {
        "risk_score": 0.0,
        "trap_type": "Liquidity Sweep Trap",
        "reasons": [],
        "main_reason": "No high-confidence trap signal",
        "buyer_seller_control": "Neutral",
        "components": {"structure_failure": 0.0, "volume_behavior": 0.0, "momentum_exhaustion": 0.0, "anomaly": 0.0},
    }

    if df_1m is None or len(df_1m) < 25 or not model:
        return default_snapshot

    frame = df_1m.copy()
    frame["price_return"] = frame["price"].pct_change().fillna(0)
    frame["volume_change"] = frame["volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    frame["volatility"] = frame["price_return"].rolling(window=10, min_periods=1).std().fillna(0)

    rolling_high = frame["price"].rolling(window=20, min_periods=5).max().shift(1)
    frame["breakout_strength"] = ((frame["price"] - rolling_high) / rolling_high.replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    frame["is_breakout"] = (frame["price"] > rolling_high).astype(int)
    frame["pv_divergence"] = ((frame["price_return"] > 0) & (frame["volume_change"] < 0)).astype(int)

    model_features = frame[
        ["price_return", "volume_change", "volatility", "breakout_strength", "is_breakout", "pv_divergence"]
    ].fillna(0)

    scores = model.predict_anomaly_scores(model_features)
    anomaly_risk_pct = float(model.risk_percentage(scores, symbol=symbol)[-1])
    anomaly_component = min(max(anomaly_risk_pct / 100.0, 0.0), 1.0)

    components, diagnostics = build_component_scores(frame[["timestamp", "price", "volume"]])
    # Final trap risk is a weighted blend of structure, volume, momentum, and anomaly context.
    weighted_risk = (
        0.34 * components["structure_failure"]
        + 0.26 * components["volume_behavior"]
        + 0.22 * components["momentum_exhaustion"]
        + 0.18 * anomaly_component
    ) * 100.0

    risk_score = float(round(max(0.0, min(weighted_risk, 100.0)), 1))
    trap_type = classify_trap_type(components, anomaly_component)

    reasons = []
    if risk_score >= CRITICAL_THRESHOLD:
        reasons = extract_trap_reasons(components, diagnostics, anomaly_component, max_reasons=3)

    return {
        "risk_score": risk_score,
        "trap_type": trap_type,
        "reasons": reasons,
        "main_reason": reasons[0]["reason"] if reasons else "No high-confidence trap signal",
        "buyer_seller_control": buyer_seller_control(frame[["timestamp", "price", "volume"]]),
        "components": {**components, "anomaly": anomaly_component},
    }


def update_risk_state(symbol: str, raw_risk_score: float):
    # Keep risk stable: smooth short-lived spikes before threshold checks.
    state = st.session_state.risk_state.setdefault(
        symbol,
        {"streak": 0, "last_smoothed_risk": 0.0, "smoothed_risk": 0.0},
    )

    if state["smoothed_risk"] == 0.0:
        smoothed_risk = raw_risk_score
    else:
        smoothed_risk = (EMA_ALPHA * raw_risk_score) + ((1 - EMA_ALPHA) * state["smoothed_risk"])

    if smoothed_risk >= CRITICAL_THRESHOLD:
        state["streak"] += 1
    else:
        state["streak"] = 0

    crossed_above = state["last_smoothed_risk"] < CRITICAL_THRESHOLD <= smoothed_risk
    state["last_smoothed_risk"] = smoothed_risk
    state["smoothed_risk"] = smoothed_risk
    return state["streak"] >= CRITICAL_STREAK_UPDATES, crossed_above, state["streak"], round(smoothed_risk, 1)


def render_reasons_panel(reasons):
    st.markdown('<div class="reasons-panel"><div style="font-size:0.72rem;color:var(--text-secondary);margin-bottom:0.45rem;">TOP TRAP REASONS</div>', unsafe_allow_html=True)
    if reasons:
        for reason in reasons:
            st.markdown(f"<div style='font-size:0.78rem; margin-bottom:0.28rem;'>• {reason['reason']} <span style='color:var(--text-secondary)'>({reason['confidence']:.1f}%)</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:0.78rem;color:var(--text-secondary);'>Monitoring for high-confidence trap reasons...</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_control_indicator(control_state: str):
    if control_state == "Buyers in Control":
        bg = "rgba(35, 209, 139, 0.12)"
        fg = "var(--bullish)"
    elif control_state == "Sellers in Control":
        bg = "rgba(248, 81, 73, 0.12)"
        fg = "var(--bearish)"
    else:
        bg = "rgba(210, 153, 34, 0.12)"
        fg = "var(--warning)"

    st.markdown(
        f"<div class='control-chip' style='background:{bg}; color:{fg};'>ORDERFLOW BIAS: {control_state.upper()}</div>",
        unsafe_allow_html=True,
    )


# Sidebar Settings
with st.sidebar:
    st.markdown("### TERMINAL SETTINGS")
    symbol_choice = st.selectbox("ACTIVE SYMBOL", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"], index=0).lower()
    refresh_rate = st.slider("SCRAPE INTERVAL (S)", 1, 5, 2)
    st.markdown("---")
    st.markdown("### SYSTEM LOG")
    if model:
        st.success("Anomaly Model: LOADED")
    else:
        st.warning("Anomaly Model: MISSING")

    st.caption(f"Critical trigger rule: {CRITICAL_STREAK_UPDATES} consecutive updates above {CRITICAL_THRESHOLD:.0f}%.")

    if st.button("RESTART FEED"):
        client.stop()
        client.start()
        st.rerun()

# Data + Snapshot
merged_1m = build_merged_1m_frame(symbol_choice)
snapshot = compute_realtime_snapshot(symbol_choice, merged_1m)
is_critical, crossed_above_70, streak_count, smoothed_risk = update_risk_state(symbol_choice, snapshot["risk_score"])
render_header(is_critical=is_critical)

curr_data = render_metrics(symbol_choice)

if crossed_above_70 and curr_data is not None:
    st.session_state.trap_history.appendleft(
        {
            "Time (UTC)": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": symbol_choice.upper(),
            "Price": round(float(curr_data["price"]), 2),
            "Trap Risk %": round(smoothed_risk, 1),
            "Trap Type": snapshot["trap_type"],
            "Main Reason": snapshot["main_reason"],
        }
    )

col_chart, col_risk = st.columns([0.7, 0.3])

with col_chart:
    st.markdown(
        """
    <div style="background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem;">
        <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 1rem; font-family: 'JetBrains Mono'">REAL-TIME TAPE</div>
    """,
        unsafe_allow_html=True,
    )
    create_advanced_chart(merged_1m)
    st.markdown("</div>", unsafe_allow_html=True)

with col_risk:
    risk_score = smoothed_risk

    st.markdown('<div class="risk-gauge-container">', unsafe_allow_html=True)
    render_risk_gauge(risk_score)

    risk_level = "LOW"
    risk_color = "var(--bullish)"
    if is_critical:
        risk_level = "CRITICAL"
        risk_color = "var(--bearish)"
    elif risk_score > 30:
        risk_level = "ELEVATED"
        risk_color = "var(--warning)"

    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 1rem; width:100%;">
            <div style="color: var(--text-secondary); font-size: 0.8rem;">STATUS</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {risk_color}; letter-spacing: 2px;">{risk_level}</div>
            <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 8px;">
                STREAK ABOVE 70%: {streak_count}/{CRITICAL_STREAK_UPDATES}
            </div>
            <div class="trap-type-badge">TRAP TYPE: {snapshot['trap_type']}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    render_control_indicator(snapshot["buyer_seller_control"])
    reasons_to_show = snapshot["reasons"] if risk_score >= CRITICAL_THRESHOLD else []
    render_reasons_panel(reasons_to_show)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### DETECTED ANOMALIES")
df_log = client.get_latest_data(symbol_choice)
if df_log is not None and len(df_log) > 0:
    disp_df = df_log.tail(10).sort_values("timestamp", ascending=False)[["timestamp", "price", "volume"]].copy()
    disp_df["Time (UTC)"] = pd.to_datetime(disp_df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    disp_df = disp_df[["Time (UTC)", "price", "volume"]]
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

st.markdown("### TRAP HISTORY (LAST 10)")
if st.session_state.trap_history:
    history_df = pd.DataFrame(list(st.session_state.trap_history))
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No trap events have crossed above 70% yet.")

# Real-time update loop
time.sleep(refresh_rate)
st.rerun()
