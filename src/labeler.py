import pandas as pd
from pathlib import Path

def create_labels(input_file=None, output_file=None):
    """
    Create trading labels based on the specified conditions.
    
    Args:
        input_file (str, optional): Path to the input CSV file. Defaults to 'data/features_pandas.csv'.
        output_file (str, optional): Path to save the output CSV file. Defaults to 'data/labeled_data.csv'.
    """
    try:
        # Set default paths relative to project root
        project_root = Path(__file__).parent.parent
        input_file = input_file or str(project_root / 'data' / 'features_pandas.csv')
        output_file = output_file or str(project_root / 'data' / 'labeled_data.csv')
        
        # Read the data
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Ensure date is in datetime format and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Initialize label column with 0
        df['label'] = 0
        
        # Check conditions for each row
        for i in range(len(df) - 3):  # We need to look ahead 3 days
            current = df.iloc[i]
            
            # Check if current row meets the initial conditions
            if current['is_breakout'] == 1 and current['volume_change'] < 0.3:  # Increased volume threshold
                # Get the close price 3 days later
                future_close = df.iloc[i + 3]['close']
                price_drop = (current['close'] - future_close) / current['close']
                
                # Check if price drops by at least 2% in the next 3 days
                if price_drop >= 0.02:  # Reduced price drop threshold
                    df.at[df.index[i], 'label'] = 1
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Successfully saved labeled data to {output_file}")
        print(f"Total rows processed: {len(df)}")
        print(f"Number of positive labels (1): {df['label'].sum()}")
        
        return df
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    create_labels()