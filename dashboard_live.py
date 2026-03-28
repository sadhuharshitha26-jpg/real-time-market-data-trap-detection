"""
MarketTrap - Clean Real-Time Dashboard with Binance Integration
A professional trading terminal interface with real-time updates from Binance API
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
import requests
import json
import websocket
import asyncio

# Import MarketTrap components
from ml_pipeline.anomaly_model import IsolationForestModel

# Configure Streamlit page
st.set_page_config(
    page_title="MarketTrap Live",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional trading terminal look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .risk-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .risk-low { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .risk-medium { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .risk-high { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-live { background-color: #10b981; animation: pulse 2s infinite; }
    .status-stale { background-color: #f59e0b; }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .metric-card {
        background: #1c2128;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #30363d;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        color: #ffffff;
    }
    .metric-card h4 {
        color: #8b949e;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-card small {
        color: #8b949e;
        font-size: 0.75rem;
    }
    /* Streamlit dark theme overrides */
    .stApp {
        background-color: #0d1117;
    }
    .stSidebar {
        background-color: #161b22;
    }
    .stSelectbox > div > div {
        background-color: #21262d;
        color: #ffffff;
    }
    .stSlider > div > div > div {
        background-color: #21262d;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class MarketData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    risk_score: float
    anomaly_score: float

class BinanceDataFetcher:
    """Real-time Binance data fetcher using REST API."""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.last_price = {}
        self.last_volume = {}
        self.current_candle = None
        
    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current ticker price from Binance."""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except Exception as e:
            st.error(f"Error fetching price: {e}")
        return None
    
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> Optional[List]:
        """Get kline/candlestick data from Binance."""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching klines: {e}")
        return None
    
    def get_24hr_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24hr ticker statistics from Binance."""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching 24hr ticker: {e}")
        return None
    
    def get_recent_trades(self, symbol: str, limit: int = 20) -> Optional[List[Dict]]:
        """Get recent trades for volume estimation."""
        try:
            url = f"{self.base_url}/api/v3/trades"
            params = {'symbol': symbol, 'limit': limit}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching trades: {e}")
        return None
    
    def estimate_volume(self, symbol: str) -> float:
        """Estimate current trading volume from recent trades."""
        trades = self.get_recent_trades(symbol, limit=10)
        if trades:
            total_volume = sum(float(trade['qty']) for trade in trades)
            return total_volume / len(trades)  # Average trade size
        return 0.0

class MarketTrapDashboard:
    def __init__(self):
        self.model = None
        self.data_buffer: List[MarketData] = []
        self.max_buffer_size = 100
        self.last_update = None
        self.is_running = False
        self.binance_fetcher = BinanceDataFetcher()
        self.current_candle = None
        self.candle_start_time = None
        
    def load_model(self):
        """Load the anomaly detection model."""
        try:
            if "model" not in st.session_state:
                st.session_state.model = IsolationForestModel.load("models/isolation_forest.pkl")
            self.model = st.session_state.model
            return True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return False
    
    def fetch_candlestick_data(self, symbol: str = 'BTCUSDT') -> Optional[MarketData]:
        """Fetch candlestick data from Binance and compute risk score."""
        try:
            # Get current price
            current_price = self.binance_fetcher.get_ticker_price(symbol)
            if current_price is None:
                return None
            
            # Get historical klines
            klines = self.binance_fetcher.get_klines(symbol, '1m', 50)
            if not klines:
                return None
            
            # Process klines into MarketData objects
            historical_data = []
            for kline in klines[:-1]:  # Exclude the most recent (incomplete) candle
                timestamp = datetime.fromtimestamp(int(kline[0]) / 1000)
                open_price = float(kline[1])
                high_price = float(kline[2])
                low_price = float(kline[3])
                close_price = float(kline[4])
                volume = float(kline[5])
                
                historical_data.append(MarketData(
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    risk_score=0.0,
                    anomaly_score=0.0
                ))
            
            # Create current candle (real-time)
            now = datetime.now()
            if self.current_candle is None or now.minute != self.candle_start_time.minute:
                # Start new candle
                self.current_candle = {
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'volume': 0.0
                }
                self.candle_start_time = now.replace(second=0, microsecond=0)
            else:
                # Update current candle
                self.current_candle['high'] = max(self.current_candle['high'], current_price)
                self.current_candle['low'] = min(self.current_candle['low'], current_price)
            
            # Get current volume
            ticker_data = self.binance_fetcher.get_24hr_ticker(symbol)
            current_volume = float(ticker_data.get('volume', 0)) if ticker_data else 0.0
            
            # Add current candle to data
            current_candle_data = MarketData(
                timestamp=self.candle_start_time,
                open=self.current_candle['open'],
                high=self.current_candle['high'],
                low=self.current_candle['low'],
                close=current_price,
                volume=current_volume,
                risk_score=0.0,
                anomaly_score=0.0
            )
            
            # Combine historical and current data
            all_data = historical_data + [current_candle_data]
            
            # Compute risk scores for all data
            if len(all_data) >= 3 and self.model:
                df = pd.DataFrame([{
                    'price': d.close,
                    'volume': d.volume,
                    'timestamp': d.timestamp
                } for d in all_data])
                
                features = self._compute_features(df)
                if len(features) > 0:
                    scores = self.model.anomaly_score(features)
                    risk_pct = self.model.risk_percentage(scores)
                    
                    # Assign risk scores to data
                    for i, data in enumerate(all_data):
                        if i < len(risk_pct):
                            data.risk_score = float(risk_pct[i])
                        if i < len(scores):
                            data.anomaly_score = float(scores[i])
            
            return current_candle_data
            
        except Exception as e:
            st.error(f"Error fetching candlestick data: {e}")
            return None
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for anomaly detection."""
        features = df.copy()
        
        # Ensure numeric types
        features['price'] = pd.to_numeric(features['price'])
        features['volume'] = pd.to_numeric(features['volume'])
        
        # Price returns
        features['price_return'] = features['price'].pct_change()
        
        # Volume changes
        features['volume_change'] = features['volume'].pct_change()
        
        # Volatility
        features['volatility'] = features['price_return'].rolling(window=10, min_periods=1).std()
        
        # Fill NA values
        features = features.fillna(0)
        
        # Ensure required columns
        required_columns = ['price_return', 'volume_change', 'volatility']
        for col in required_columns:
            if col not in features.columns:
                features[col] = 0
        
        return features[required_columns]
    
    def update_data_buffer(self, new_data: MarketData):
        """Update the data buffer with new market data."""
        self.data_buffer.append(new_data)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
        self.last_update = datetime.now()
    
    def get_risk_level(self, risk_score: float) -> str:
        """Get risk level category."""
        if risk_score < 30:
            return "Low"
        elif risk_score < 70:
            return "Medium"
        else:
            return "High"
    
    def get_risk_color(self, risk_score: float) -> str:
        """Get risk color for styling."""
        if risk_score < 30:
            return "risk-low"
        elif risk_score < 70:
            return "risk-medium"
        else:
            return "risk-high"

def create_candlestick_chart(data_buffer: List[MarketData], symbol: str) -> go.Figure:
    """Create candlestick chart with smooth real-time updates."""
    if not data_buffer:
        # Create empty chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(f"{symbol} Candlestick Chart", "Trap Risk %"),
            row_heights=[0.65, 0.35]
        )
        fig.update_layout(height=600, showlegend=False)
        return fig
    
    # Extract data
    timestamps = [d.timestamp for d in data_buffer]
    opens = [d.open for d in data_buffer]
    highs = [d.high for d in data_buffer]
    lows = [d.low for d in data_buffer]
    closes = [d.close for d in data_buffer]
    volumes = [d.volume for d in data_buffer]
    risk_scores = [d.risk_score for d in data_buffer]
    
    # Create figure with animation for smooth updates
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"{symbol} Candlestick Chart", "Trap Risk %"),
        row_heights=[0.65, 0.35]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price",
            increasing=dict(line=dict(color="#00ff88"), fillcolor="#00ff88"),
            decreasing=dict(line=dict(color="#ff4444"), fillcolor="#ff4444")
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#00ff88' if close >= open else '#ff4444' for close, open in zip(closes, opens)]
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=volumes,
            name="Volume",
            marker_color=colors,
            opacity=0.3,
            yaxis="y2",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Risk percentage line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=risk_scores,
            name="Risk %",
            line=dict(color="#ff9500", width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 149, 0, 0.2)',
            hovertemplate='%{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Risk level zones
    fig.add_hrect(
        y0=0, y1=30, fillcolor="rgba(16, 185, 129, 0.1)", 
        line_width=0, row=2, col=1, annotation_text="Low Risk"
    )
    fig.add_hrect(
        y0=30, y1=70, fillcolor="rgba(245, 158, 11, 0.1)", 
        line_width=0, row=2, col=1, annotation_text="Medium Risk"
    )
    fig.add_hrect(
        y0=70, y1=100, fillcolor="rgba(239, 68, 68, 0.1)", 
        line_width=0, row=2, col=1, annotation_text="High Risk"
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Risk %", range=[0, 100], row=2, col=1)
    
    # Update layout for professional trading look with smooth updates
    fig.update_layout(
        height=600,
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='#30363d',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#30363d',
            showgrid=True
        ),
        yaxis2=dict(
            gridcolor='#30363d',
            showgrid=False
        ),
        yaxis3=dict(
            gridcolor='#30363d',
            showgrid=True
        ),
        # Smooth transition settings
        transition=dict(duration=200, easing="cubic-in-out")
    )
    
    return fig

def main():
    """Main dashboard application."""
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = MarketTrapDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Header
    st.markdown('<h1 class="main-header">📊 MarketTrap Live</h1>', unsafe_allow_html=True)
    st.markdown("Real-time market anomaly detection and risk analysis")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        symbol = st.selectbox(
            "Trading Pair",
            ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"],
            index=0
        )
        
        auto_refresh = st.toggle("🔄 Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 2)
        
        st.markdown("---")
        
        # Binance API status
        st.subheader("📡 Binance API Status")
        if dashboard.last_update:
            time_since_update = (datetime.now() - dashboard.last_update).total_seconds()
            if time_since_update < 10:
                st.markdown('<span class="status-indicator status-live"></span>Connected', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-indicator status-stale"></span>Stale', unsafe_allow_html=True)
            st.write(f"Last update: {time_since_update:.1f}s ago")
        else:
            st.markdown('<span class="status-indicator status-stale"></span>Connecting...', unsafe_allow_html=True)
        
        # Data buffer info
        st.write(f"Data points: {len(dashboard.data_buffer)}")
        st.write(f"Candlesticks: {len(dashboard.data_buffer)}")
    
    # Load model
    if not dashboard.load_model():
        st.error("Cannot start dashboard without model")
        return
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    # Metrics row
    with col1:
        if dashboard.data_buffer:
            latest = dashboard.data_buffer[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h4>Current Price</h4>
                <h2>${latest.close:,.2f}</h2>
                <small>O: ${latest.open:,.2f} H: ${latest.high:,.2f} L: ${latest.low:,.2f}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if dashboard.data_buffer:
            latest = dashboard.data_buffer[-1]
            risk_class = dashboard.get_risk_color(latest.risk_score)
            st.markdown(f"""
            <div class="risk-metric {risk_class}">
                <h4>Risk Score</h4>
                <h2>{latest.risk_score:.1f}%</h2>
                <p>{dashboard.get_risk_level(latest.risk_score)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if dashboard.data_buffer:
            latest = dashboard.data_buffer[-1]
            price_change = latest.close - latest.open
            price_change_pct = (price_change / latest.open) * 100 if latest.open > 0 else 0
            change_color = "#00ff88" if price_change >= 0 else "#ff4444"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Candle Change</h4>
                <h2 style="color: {change_color}">{price_change_pct:+.2f}%</h2>
                <small>{price_change:+.2f} USD</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if dashboard.data_buffer:
            latest = dashboard.data_buffer[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h4>Volume</h4>
                <h2>{latest.volume:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Chart container - use session state to persist chart
    if 'chart_figure' not in st.session_state:
        st.session_state.chart_figure = None
    
    chart_placeholder = st.empty()
    
    # Real-time update function
    def update_dashboard():
        """Update dashboard with new candlestick data without recreating chart."""
        new_data = dashboard.fetch_candlestick_data(symbol)
        if new_data:
            # Update data buffer with full candlestick data
            klines = dashboard.binance_fetcher.get_klines(symbol, '1m', 50)
            if klines:
                dashboard.data_buffer.clear()
                for kline in klines:
                    timestamp = datetime.fromtimestamp(int(kline[0]) / 1000)
                    open_price = float(kline[1])
                    high_price = float(kline[2])
                    low_price = float(kline[3])
                    close_price = float(kline[4])
                    volume = float(kline[5])
                    
                    candle_data = MarketData(
                        timestamp=timestamp,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        close=close_price,
                        volume=volume,
                        risk_score=0.0,
                        anomaly_score=0.0
                    )
                    dashboard.data_buffer.append(candle_data)
                
                # Update current candle with real-time data
                if dashboard.data_buffer:
                    dashboard.data_buffer[-1] = new_data
                
                # Limit buffer size
                if len(dashboard.data_buffer) > dashboard.max_buffer_size:
                    dashboard.data_buffer = dashboard.data_buffer[-dashboard.max_buffer_size:]
            
            # Create or update chart figure
            fig = create_candlestick_chart(dashboard.data_buffer, symbol)
            st.session_state.chart_figure = fig
    
    # Initial update
    update_dashboard()
    
    # Display chart (will persist across reruns)
    if st.session_state.chart_figure:
        chart_placeholder.plotly_chart(st.session_state.chart_figure, use_container_width=True)
    
    # Real-time updates
    if auto_refresh:
        # Use Streamlit's built-in rerun for real-time updates
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
