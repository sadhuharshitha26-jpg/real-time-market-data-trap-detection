import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from feature_engineering.trap_features import compute_trap_features
from ml_pipeline.anomaly_model import IsolationForestModel
from risk_inference.trap_risk import compute_trap_risk

st.set_page_config(page_title="MarketTrap", layout="wide")

st.title("MarketTrap – Whale Trap Risk Monitor")

symbol = st.selectbox(
    "Select Cryptocurrency",
    ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
)

@st.cache_data(ttl=60)
def load_data(symbol: str):
    """Load latest OHLCV data for the symbol."""
    # Simulated data loader; replace with real CSV/DB/API call
    np.random.seed(42)
    periods = 100
    base_price = {"BTC-USD": 43000, "ETH-USD": 2600, "SOL-USD": 105, "BNB-USD": 310, "XRP-USD": 0.62}[symbol]
    timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq="1min")
    price = np.cumprod(1 + np.random.randn(periods) * 0.001) * base_price
    volume = np.random.randint(1_000_000, 10_000_000, size=periods)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": price,
        "high": price * (1 + np.random.rand(periods) * 0.002),
        "low": price * (1 - np.random.rand(periods) * 0.002),
        "close": price,
        "volume": volume,
    })
    return df

def run_pipeline(df: pd.DataFrame):
    """Run trap risk pipeline on the dataframe."""
    features = compute_trap_features(df)
    model = IsolationForestModel.load("models/isolation_forest.pkl")
    anomaly_scores = model.anomaly_score(features)
    result = compute_trap_risk(features, anomaly_score=float(anomaly_scores[-1]))
    return result, features, anomaly_scores

data = load_data(symbol)

result, features, anomaly_scores = run_pipeline(data)

# 📈 Combined chart (price + risk)
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Price", "Trap Risk %"),
    row_heights=[0.6, 0.4]
)

fig.add_trace(
    go.Scatter(x=data["timestamp"], y=data["close"], name="Price", line=dict(color="#1f77b4")),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=data["timestamp"],
        y=[result["trap_risk_score"]] * len(data),
        name="Current Risk",
        line=dict(color="#ff7f0e", dash="dash")
    ),
    row=2, col=1
)

fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Risk %", range=[0, 100], row=2, col=1)
fig.update_layout(height=600, showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# 🚨 Risk summary
st.metric(
    label="Current Trap Risk",
    value=f"{result['trap_risk_score']:.1f}%",
    delta=result["risk_level"]
)

# 🧠 Explanation
st.subheader("Why this looks risky")
for reason in result["top_3_reasons"]:
    st.write("•", reason)

if result["invalidated_by"]:
    st.subheader("What would invalidate this trap")
    for inv in result["invalidated_by"]:
        st.write("•", inv)

# 📊 Component breakdown
st.subheader("Component breakdown")
st.json(result["components"])
