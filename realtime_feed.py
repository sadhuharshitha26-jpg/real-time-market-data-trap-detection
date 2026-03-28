"""
Real-time crypto trade feed via CryptoCompare WebSocket.
Buffers ticks in memory for downstream feature engine.
"""

import json
import time
import websocket
from collections import deque
from typing import Deque, Dict, List, Optional

import pandas as pd

# Store last 500 ticks per symbol
tick_buffers: Dict[str, Deque[Dict]] = {}
ws_app: Optional[websocket.WebSocketApp] = None
last_tick_time: Dict[str, float] = {}
connection_time: Optional[float] = None

def on_message(ws: websocket.WebSocketApp, message: str) -> None:
    """Handle incoming WebSocket messages."""
    data = json.loads(message)

    if data.get("TYPE") == "5":  # Trade message
        symbol = data.get("FROMSYMBOL", "")
        if not symbol:
            return

        if symbol not in tick_buffers:
            tick_buffers[symbol] = deque(maxlen=500)

        tick = {
            "timestamp": time.time(),
            "price": float(data.get("PRICE", 0)),
            "volume": float(data.get("VOLUME", 0)),
            "symbol": symbol,
        }
        tick_buffers[symbol].append(tick)
        last_tick_time[symbol] = tick["timestamp"]

def on_open(ws: websocket.WebSocketApp) -> None:
    """Subscribe to trade streams on open."""
    global connection_time
    connection_time = time.time()
    subs = {
        "action": "SubAdd",
        "subs": [
            "5~CCCAGG~BTC~USD",
            "5~CCCAGG~ETH~USD",
            "5~CCCAGG~SOL~USD",
            "5~CCCAGG~BNB~USD",
            "5~CCCAGG~XRP~USD",
        ],
    }
    ws.send(json.dumps(subs))
    print("Subscribed to CryptoCompare trade streams.")

def on_error(ws: websocket.WebSocketApp, error: Exception) -> None:
    """Log WebSocket errors."""
    print(f"WebSocket error: {error}")

def on_close(ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
    """Handle WebSocket close and trigger reconnect."""
    print("WebSocket closed. Will attempt to reconnect in 5 seconds...")
    time.sleep(5)
    start_ws(ws.url.split("?api_key=")[-1])  # Re-extract API key from URL

def start_ws(api_key: str) -> None:
    """Start the WebSocket connection with retry/backoff."""
    global ws_app
    ws_url = f"wss://streamer.cryptocompare.com/v2?api_key={api_key}"
    ws_app = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
    )
    ws_app.run_forever(ping_interval=20, ping_timeout=10)

def get_latest_ticks(symbol: str, min_ticks: int = 20) -> Optional[pd.DataFrame]:
    """Return the latest ticks for a symbol as a DataFrame."""
    buffer = tick_buffers.get(symbol)
    if not buffer or len(buffer) < min_ticks:
        return None
    return pd.DataFrame(list(buffer))

def build_ohlcv_from_ticks(ticks_df: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """
    Convert raw ticks into OHLCV candles.
    
    Args:
        ticks_df: DataFrame with columns timestamp, price, volume, symbol
        window_seconds: Candle interval in seconds (default 60s)
    """
    if ticks_df is None or ticks_df.empty:
        return pd.DataFrame()

    # Round timestamps to nearest window
    ticks_df["candle_time"] = (ticks_df["timestamp"] // window_seconds) * window_seconds
    grouped = ticks_df.groupby("candle_time")

    ohlcv = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    ohlcv["timestamp"] = pd.to_datetime(ohlcv["candle_time"], unit="s")
    return ohlcv[["timestamp", "open", "high", "low", "close", "volume"]]

def get_latest_ohlcv(symbol: str, min_ticks: int = 20, window_seconds: int = 60) -> Optional[pd.DataFrame]:
    """Convenient helper: get latest OHLCV for a symbol."""
    ticks_df = get_latest_ticks(symbol, min_ticks=min_ticks)
    if ticks_df is None:
        return None
    return build_ohlcv_from_ticks(ticks_df, window_seconds=window_seconds)

def get_connection_status() -> Dict[str, any]:
    """Return connection health and tick freshness for UI."""
    now = time.time()
    status = {
        "connected": ws_app is not None and connection_time is not None,
        "connection_age_seconds": now - connection_time if connection_time else None,
        "symbols": {}
    }
    for sym in tick_buffers.keys():
        last_seen = last_tick_time.get(sym)
        status["symbols"][sym] = {
            "buffer_size": len(tick_buffers.get(sym, [])),
            "last_tick_age_seconds": now - last_seen if last_seen else None,
            "fresh": last_seen and (now - last_seen) < 30  # fresh if tick < 30s ago
        }
    return status
