import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
from ml_pipeline.anomaly_model import IsolationForestModel
import joblib

def score_traps(features_file=None, model_path=None, output_file=None):
    """
    Score potential market traps using the trained model.
    
    Args:
        features_file (str, optional): Path to the features CSV file. Defaults to 'data/features_pandas.csv'.
        model_path (str, optional): Path to the trained model. Defaults to 'models/isolation_forest.pkl'.
        output_file (str, optional): Path to save the output CSV file. Defaults to 'outputs/trap_scores.csv'.
    """
    try:
        # Set default paths
        project_root = Path(__file__).parent.parent
        features_file = features_file or str(project_root / 'data' / 'features_pandas.csv')
        model_path = model_path or str(project_root / 'models' / 'isolation_forest.pkl')
        output_file = output_file or str(project_root / 'outputs' / 'trap_scores.csv')
        
        # Create outputs directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load the data
        print(f"Reading features from {features_file}...")
        df = pd.read_csv(features_file)
        
        # Ensure date is in datetime format and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Define features (same as training)
        features = [
            'price_return', 'volume_change', 'volatility',
            'breakout_strength', 'is_breakout', 'pv_divergence'
        ]
        
        # Load the model
        print("Loading model...")
        model = IsolationForestModel.load(model_path)
        
        # Calculate anomaly scores and risk percentages
        print("Calculating anomaly scores...")
        scores = model.anomaly_score(df[features])
        risk_pct = model.risk_percentage(scores)
        
        # Add results to the dataframe
        df['anomaly_score'] = scores
        df['trap_risk'] = risk_pct
        
        # Classify risk levels
        df['risk_level'] = pd.cut(
            df['trap_risk'],
            bins=[0, 30, 70, 100],
            labels=['Normal', 'Suspicious', 'Potential Market Trap']
        )
        
        # Reorder columns to have date and scores first
        columns = ['date', 'trap_risk', 'risk_level'] + [col for col in df.columns if col not in ['date', 'trap_risk', 'risk_level']]
        df = df[columns]
        
        # Save the results
        df.to_csv(output_file, index=False)
        print(f"Successfully saved trap scores to {output_file}")
        print("\nRisk Category Distribution:")
        print(df['risk_level'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    score_traps()