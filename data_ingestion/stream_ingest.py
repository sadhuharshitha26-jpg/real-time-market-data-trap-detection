"""
Real-time market data streaming ingestion.

This module simulates a real-time market data stream for development and testing.
In production, this would connect to Apache Kafka or another streaming platform.

Example Kafka Production Setup:
    from kafka import KafkaConsumer, KafkaProducer
    consumer = KafkaConsumer('market_data', bootstrap_servers='kafka:9092')
    for message in consumer:
        process_message(message)
"""

import time
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketDataStream:
    """Simulates a real-time market data stream with configurable parameters."""
    
    def __init__(self, symbols: list = None, batch_size: int = 10, 
                 interval: float = 1.0, max_messages: int = None):
        """
        Initialize the market data stream.
        
        Args:
            symbols: List of trading pair symbols (e.g., ['BTC-USD', 'ETH-USD'])
            batch_size: Number of messages to yield in each batch per symbol
            interval: Time in seconds between batches
            max_messages: Maximum number of messages to generate per symbol (None for infinite)
        """
        self.symbols = symbols or ['BTC-USD']
        self.batch_size = batch_size
        self.interval = interval
        self.max_messages = max_messages
        self.message_count = {symbol: 0 for symbol in self.symbols}
        self.historical_data = {}
        self._load_historical_data()
        
    def _load_historical_data(self):
        """Load historical data to make simulation more realistic."""
        data_dir = Path(__file__).parent.parent / 'data'
        historical_file = data_dir / 'historical_market_data.csv'
        
        try:
            if historical_file.exists():
                self.historical_data = pd.read_csv(historical_file)
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
                print(f"Loaded {len(self.historical_data)} historical records")
            else:
                print("No historical data found, using synthetic data")
                self.historical_data = None
        except Exception as e:
            print(f"Error loading historical data: {e}")
            self.historical_data = None
    
    def _generate_market_data(self, symbol: str) -> Dict[str, Any]:
        """Generate a single market data message for the given symbol."""
        current_time = datetime.utcnow()

        # Get base price and volume from historical data if available
        if symbol in self.historical_data:
            hist_row = self.historical_data.sample(1).iloc[0]
            price = hist_row['close']
            volume = hist_row['volume']
        else:
            # Default values if no historical data
            price = random.uniform(1000, 100000)
            volume = random.uniform(1, 1000)

        # Add some randomness to make it look like live data
        price *= (1 + random.uniform(-0.005, 0.005))
        volume *= (1 + random.uniform(-0.1, 0.1))

        return {
            'symbol': symbol,
            'timestamp': current_time.isoformat() + 'Z',
            'price': round(price, 2),
            'volume': round(volume, 2),
            'exchange': 'simulated',
            'sequence': self.message_count[symbol]
        }

    def stream(self) -> Iterator[Dict[str, Any]]:
        """
        Generate a stream of market data messages for all symbols.

        Yields:
            Dictionary mapping symbols to their respective batch of market data messages
        """
        while True:
            try:
                # Check if we've reached max messages for all symbols
                if self.max_messages is not None and all(
                    count >= self.max_messages for count in self.message_count.values()
                ):
                    logger.info("Reached maximum message count for all symbols")
                    break

                start_time = time.time()
                batch = {symbol: [] for symbol in self.symbols}

                # Generate messages for each symbol
                for symbol in self.symbols:
                    if self.max_messages is not None and self.message_count[symbol] >= self.max_messages:
                        continue

                    for _ in range(self.batch_size):
                        message = self._generate_market_data(symbol)
                        batch[symbol].append(message)
                        self.message_count[symbol] += 1

                        if self.max_messages is not None and self.message_count[symbol] >= self.max_messages:
                            break

                # Yield only if we have data for at least one symbol
                if any(batch.values()):
                    yield batch

                    # Calculate sleep time to maintain desired interval
                    elapsed = time.time() - start_time
                    sleep_time = max(0, self.interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Stream interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in stream: {str(e)}")
                time.sleep(1)  # Prevent tight loop on error
            
def process_stream(symbols: list = None, batch_size: int = 10, 
                  interval: float = 1.0, max_messages: int = None):
    """
    Process the market data stream with the given parameters.
    
    Args:
        symbols: List of trading pair symbols (e.g., ['BTC-USD', 'ETH-USD'])
        batch_size: Number of messages per batch per symbol
        interval: Time in seconds between batches
        max_messages: Maximum number of messages to generate per symbol
    """
    stream = MarketDataStream(
        symbols=symbols,
        batch_size=batch_size,
        interval=interval,
        max_messages=max_messages
    )
    
    try:
        for batch in stream.stream():
            # Process each symbol's batch
            for symbol, messages in batch.items():
                if messages:  # Only process if we have messages
                    logger.info(f"Processed {len(messages)} messages for {symbol}")
                    # Here you would typically send to a processing queue
                    # For now, just log the first message of each batch
                    if messages:
                        logger.debug(f"Sample message for {symbol}: {messages[0]}")
    except KeyboardInterrupt:
        logger.info("Stream processing stopped by user")
    except Exception as e:
        logger.error(f"Error processing stream: {str(e)}")
        raise
    
if __name__ == "__main__":
    # Example usage with parameters
    process_stream(
        symbols=['BTC-USD'],
        batch_size=5,
        interval=2.0,
        max_messages=20
    )