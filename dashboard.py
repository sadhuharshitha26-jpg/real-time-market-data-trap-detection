"""
MarketTrap Dashboard
Real-Time Multi-Crypto Trap Risk Monitor

• Streamlit-based UI
• Multi-crypto support
• Reads from outputs/trap_scores.csv
• Falls back to simulated data if file is missing
• AI-friendly visualization (Risk % clearly visible)

In production, this would connect to Apache Kafka.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MarketTrap",
    page_icon="📊",
    layout="wide"
)

# ---------------- STYLES ----------------
st.markdown("""
<style>
.metric {
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}
.high { background-color: #ffebee; color: #c62828; }
.medium { background-color: #fff8e1; color: #f9a825; }
.normal { background-color: #e8f5e9; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def risk_label(score):
    if score >= 0.7:
        return "🔴 High Risk", "high"
    elif score >= 0.4:
        return "🟡 Risky", "medium"
    return "🟢 Normal", "normal"


def load_data():
    """Load trap_scores.csv or fallback to simulated data"""
    path = "outputs/trap_scores.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)

        # expected minimal columns
        required = {"timestamp", "symbol", "price", "volume", "risk_score"}
        if not required.issubset(df.columns):
            st.warning("CSV format unexpected. Using demo data.")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

    # ---------- FALLBACK DEMO DATA ----------
    now = datetime.utcnow()
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]

    rows = []
    for sym in symbols:
        base = np.random.randint(1500, 30000)
        for i in range(30):
            rows.append({
                "timestamp": now - timedelta(seconds=30-i),
                "symbol": sym,
                "price": base + np.random.randn() * base * 0.01,
                "volume": np.random.randint(1e6, 1e8),
                "risk_score": np.clip(
                    0.3 + 0.4 * np.sin(i / 5) + np.random.randn() * 0.1,
                    0, 1
                )
            })

    return pd.DataFrame(rows)


# ---------------- LOAD DATA ----------------
df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Data Source")

symbols = sorted(df["symbol"].unique())
selected_symbol = st.sidebar.selectbox("Select Cryptocurrency", symbols)

st.sidebar.header("Display Settings")
auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)", 1, 10, 2, disabled=not auto_refresh
)

# ---------------- FILTER DATA ----------------
df_sym = df[df["symbol"] == selected_symbol].sort_values("timestamp")

# ---------------- HEADER ----------------
st.title("📊 Real-Time Multi-Crypto Trap Risk Monitor")

# ---------------- STATUS CARD ----------------
latest = df_sym.iloc[-1]
label, css = risk_label(latest["risk_score"])

st.markdown(
    f"""
    <div class="metric {css}">
        {label}<br>
        Price: ${latest['price']:,.2f}<br>
        Volume: {latest['volume']:,.0f}<br>
        Risk Score: {latest['risk_score']*100:.1f}%
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- CHARTS ----------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    row_heights=[0.6, 0.4],
    subplot_titles=[
        f"{selected_symbol} Price & Volume",
        "Trap Risk Percentage"
    ]
)

# Price
fig.add_trace(
    go.Scatter(
        x=df_sym["timestamp"],
        y=df_sym["price"],
        name="Price",
        line=dict(color="#1e88e5", width=2)
    ),
    row=1, col=1
)

# Volume
fig.add_trace(
    go.Bar(
        x=df_sym["timestamp"],
        y=df_sym["volume"],
        name="Volume",
        opacity=0.4
    ),
    row=1, col=1
)

# Risk %
fig.add_trace(
    go.Scatter(
        x=df_sym["timestamp"],
        y=df_sym["risk_score"] * 100,
        name="Risk %",
        line=dict(color="#e53935", width=2),
        fill="tozeroy"
    ),
    row=2, col=1
)

# Risk threshold
fig.add_hline(
    y=70, line_dash="dash", line_color="red",
    row=2, col=1
)

fig.update_layout(
    height=800,
    hovermode="x unified",
    showlegend=False
)

fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Risk %", range=[0, 100], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- DEBUG (COLLAPSIBLE) ----------------
with st.expander("🔍 Debug Info"):
    st.write(df_sym.tail())
    st.write("Rows:", df_sym.shape[0])

# ---------------- AUTO REFRESH ----------------
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
