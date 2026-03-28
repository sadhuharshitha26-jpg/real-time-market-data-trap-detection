import pandas as pd
import os
from pathlib import Path

def compute_features(input_file=None, output_file=None):
    """
    Compute technical features from OHLCV data and save to CSV.
    
    Args:
        input_file (str, optional): Path to the input CSV file. Defaults to 'data/btc_usd.csv'.
        output_file (str, optional): Path to save the output CSV file. Defaults to 'data/features_pandas.csv'.
    """
    try:
        # Set default paths relative to project root
        project_root = Path(__file__).parent.parent
        input_file = input_file or str(project_root / 'data' / 'btc_usd.csv')
        output_file = output_file or str(project_root / 'data' / 'features_pandas.csv')
        
        # Read the data
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Ensure date is in datetime format and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 1. Price return
        df['price_return'] = df['close'].pct_change()
        
        # 2. Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # 3. Volatility
        df['volatility'] = (df['high'] - df['low']) / df['open']
        
        # 4. 20-day rolling maximum of Close
        df['rolling_max_20'] = df['close'].rolling(window=20, min_periods=1).max()
        
        # 5. Breakout strength
        df['breakout_strength'] = (df['close'] - df['rolling_max_20']) / df['rolling_max_20']
        
        # 6. Is breakout (1 if Close > previous rolling_max_20, else 0)
        df['is_breakout'] = (df['close'] > df['rolling_max_20'].shift(1)).astype(int)
        
        # 7. Price-Volume divergence
        df['pv_divergence'] = ((df['price_return'] > 0) & (df['volume_change'] < 0)).astype(int)
        
        # Fill missing values with 0
        df.fillna(0, inplace=True)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Successfully saved features to {output_file}")
        print(f"Total rows processed: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    compute_features()