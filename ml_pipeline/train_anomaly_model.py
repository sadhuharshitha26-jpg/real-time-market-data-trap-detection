import pandas as pd
import numpy as np
from pathlib import Path
from .anomaly_model import IsolationForestModel

def train_trap_model(input_file=None, model_path=None):
    """
    Train an anomaly detection model for market trap identification using Isolation Forest.
    The model identifies unusual market patterns that may indicate traps in an unsupervised manner.
    
    Args:
        input_file (str, optional): Path to the historical market data CSV file. 
            Defaults to 'data/historical_market_data.csv'.
        model_path (str, optional): Path to save the trained model. 
            The scaler will be saved with '_scaler' suffix in the same directory.
            Defaults to 'models/isolation_forest.pkl'.
    """
    try:
        # Set default paths
        project_root = Path(__file__).parent.parent
        input_file = input_file or str(project_root / 'data' / 'historical_market_data.csv')
        model_path = model_path or str(project_root / 'models' / 'isolation_forest.pkl')
        
        # Read the historical market data
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Ensure date is in datetime format and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Initialize and train the model
        print("Training Isolation Forest model for anomaly detection...")
        model = IsolationForestModel(contamination='auto', random_state=42)
        model.fit(df)
        
        # Get anomaly scores and risk percentages
        scores = model.anomaly_score(df)
        risk_pct = model.risk_percentage(scores)
        
        print("\nAnomaly Detection Summary:")
        print(f"Trained on {len(df)} samples")
        print(f"Mean anomaly score: {scores.mean():.4f}")
        print(f"Min anomaly score: {scores.min():.4f}")
        print(f"Max anomaly score: {scores.max()}")
        print(f"Mean risk percentage: {risk_pct.mean():.2f}%")
        
        # Save the trained model
        model.save(model_path)
        print(f"\nModel and scaler saved to {model_path}")
        print("Note: The scaler is saved with '_scaler' suffix in the same directory.")
        
        return model
        
    except Exception as e:
        import traceback
        print(f"Error during model training: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    model = train_trap_model()
    if model is not False:
        print("\nModel training completed successfully!")