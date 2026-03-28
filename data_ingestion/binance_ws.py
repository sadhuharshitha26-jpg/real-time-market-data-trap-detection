"""
Binance WebSocket client for real-time crypto trade and ticker data.
Handles connection management, buffering, and data processing.
"""

import json
import time
import threading
import logging
from collections import deque
from typing import Dict, List, Optional, Deque
import websocket
import pandas as pd

logger = logging.getLogger(__name__)

class BinanceWSClient:
    """Robust Binance WebSocket client for streaming market data."""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = [s.lower() for s in (symbols or ["btcusdt"])]
        self.base_url = "wss://stream.binance.com:9443"
        self.tick_buffers: Dict[str, Deque[Dict]] = {s: deque(maxlen=1000) for s in self.symbols}
        self.last_price: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.last_update: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.ws: Optional[websocket.WebSocketApp] = None
        self.is_running = False
        self.reconnect_delay = 5
        
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            # Handle Ticker data (24hr ticker or Mini-Ticker)
            # We'll use individual symbol streams for simplicity
            stream_name = data.get("stream", "")
            msg_data = data.get("data", {})
            
            if "@ticker" in stream_name:
                symbol = msg_data.get("s", "").lower()
                if symbol in self.symbols:
                    tick = {
                        "timestamp": msg_data["E"] / 1000,
                        "price": float(msg_data["c"]),
                        "high": float(msg_data["h"]),
                        "low": float(msg_data["l"]),
                        "volume": float(msg_data["v"]),
                        "quote_volume": float(msg_data["q"]),
                        "symbol": symbol
                    }
                    self.tick_buffers[symbol].append(tick)
                    self.last_price[symbol] = tick["price"]
                    self.last_update[symbol] = tick["timestamp"]
            
            elif "@trade" in stream_name:
                symbol = msg_data.get("s", "").lower()
                if symbol in self.symbols:
                    # You could process individual trades here for higher resolution
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing WS message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.is_running:
            logger.info(f"Attempting to reconnect in {self.reconnect_delay} seconds...")
            time.sleep(self.reconnect_delay)
            self._connect()

    def _on_open(self, ws):
        logger.info(f"WebSocket connected for symbols: {self.symbols}")
        # Subscriptions happen via the URL in this implementation, 
        # but could also be done via send() commands.

    def _connect(self):
        # Build stream path
        # Example: btcbusd@ticker/ethbusd@ticker
        streams = "/".join([f"{s}@ticker" for s in self.symbols])
        url = f"{self.base_url}/stream?streams={streams}"
        
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        self.ws.run_forever(ping_interval=20, ping_timeout=10)

    def start(self):
        """Start the WebSocket connection in a background thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._connect, daemon=True)
        self.thread.start()
        logger.info("Binance WS Thread started.")

    def stop(self):
        """Stop the WebSocket connection."""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("Binance WS client stopped.")

    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return the buffered data for a symbol as a DataFrame."""
        symbol = symbol.lower()
        buffer = self.tick_buffers.get(symbol)
        if not buffer:
            return None
        
        # Lock not strictly needed due to GIL and deque thread-safety for append/pop
        # but list(buffer) might have slight race conditions during heavy updates
        return pd.DataFrame(list(buffer))

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    client = BinanceWSClient(symbols=["btcusdt", "ethusdt"])
    client.start()
    
    try:
        while True:
            time.sleep(5)
            df = client.get_latest_data("btcusdt")
            if df is not None and not df.empty:
                print(f"Latest BTC Price: {df['price'].iloc[-1]} | Ticks: {len(df)}")
            else:
                print("Connecting...")
    except KeyboardInterrupt:
        client.stop()
