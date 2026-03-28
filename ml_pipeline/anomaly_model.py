"""
Isolation Forest Model for Anomaly Detection

This module implements an Isolation Forest-based anomaly detection model
that can process multiple cryptocurrency symbols with shared model weights.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class IsolationForestModel:
    """
    Wrapper for scikit-learn's IsolationForest with additional functionality
    for cryptocurrency market data.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        window_size: int = 100
    ):
        """
        Initialize the Isolation Forest model.
        
        Args:
            contamination: Expected proportion of outliers in the data
            n_estimators: Number of base estimators in the ensemble
            random_state: Random seed for reproducibility
            window_size: Size of the rolling window for min-max normalization
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.window_size = window_size
        self.score_window = {}  # Store score windows per symbol
        self.feature_columns = None
        self.is_fitted = False
        
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for model input.
        
        Args:
            df: Input DataFrame with OHLCV and derived features
            
        Returns:
            Tuple of (features_array, feature_columns)
        """
        # Select only numeric columns and drop any remaining NaNs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = df[numeric_cols].dropna(axis=1)

        if self.feature_columns is None:
            self.feature_columns = features.columns.tolist()
        else:
            # Align features to the saved training columns
            missing = [col for col in self.feature_columns if col not in features.columns]
            for col in missing:
                features[col] = 0
            extra = [col for col in features.columns if col not in self.feature_columns]
            if extra:
                features = features.drop(columns=extra)
            features = features[self.feature_columns]
        # Enforce the 6 features the scaler expects
        required_6 = ['price_return', 'volume_change', 'volatility', 'breakout_strength', 'is_breakout', 'pv_divergence']
        for col in required_6:
            if col not in features.columns:
                features[col] = 0
        features = features[required_6]
        self.feature_columns = required_6

        return features.values, self.feature_columns
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> 'IsolationForestModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Training data (DataFrame or numpy array)
            y: Ignored (for scikit-learn compatibility)
            
        Returns:
            self: Returns an instance of self
        """
        if isinstance(X, pd.DataFrame):
            X, _ = self._prepare_features(X)
            
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        return self
    
    def predict_anomaly_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict anomaly scores for the input data.
        
        Args:
            X: Input data (DataFrame or numpy array)
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        if isinstance(X, pd.DataFrame):
            X, _ = self._prepare_features(X)
            
        X_scaled = self.scaler.transform(X)
        return -self.model.score_samples(X_scaled)  # Convert to positive (higher = more anomalous)

    def anomaly_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Backward-compatible alias for predict_anomaly_scores."""
        return self.predict_anomaly_scores(X)
    
    def compute_risk_percentage(
        self,
        scores: np.ndarray,
        symbol: str
    ) -> np.ndarray:
        """
        Convert anomaly scores to risk percentages using rolling window min-max normalization.
        
        Args:
            scores: Array of anomaly scores
            symbol: Cryptocurrency symbol for maintaining separate windows
            
        Returns:
            Array of risk percentages (0-100)
        """
        if symbol not in self.score_window:
            self.score_window[symbol] = []
            
        risk_scores = []
        
        for score in scores:
            # Update score window
            self.score_window[symbol].append(score)
            if len(self.score_window[symbol]) > self.window_size:
                self.score_window[symbol].pop(0)
                
            # Get current window
            window = np.array(self.score_window[symbol])
            
            # Calculate min and max in the window
            min_score = np.min(window)
            max_score = np.max(window)
            score_range = max_score - min_score
            
            # Avoid division by zero
            if score_range > 0:
                normalized = (score - min_score) / score_range
            else:
                normalized = 0.5  # Default to middle if no variation
                
            # Convert to percentage (0-100)
            risk_scores.append(min(max(normalized * 100, 0), 100))
            
        return np.array(risk_scores)

    def risk_percentage(self, scores: np.ndarray, symbol: str = "BTC-USD") -> np.ndarray:
        """Backward-compatible risk percentage helper with default symbol."""
        return self.compute_risk_percentage(scores, symbol)
    
    def predict(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Predict anomaly scores and compute risk percentages for input data.
        
        Args:
            df: Input DataFrame with features
            symbol: Cryptocurrency symbol
            
        Returns:
            DataFrame with timestamp, symbol, risk_percentage, and anomaly_score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        # Make a copy to avoid modifying the input
        df = df.copy()
        
        # Get anomaly scores
        anomaly_scores = self.predict_anomaly_scores(df)
        
        # Compute risk percentages
        risk_percentages = self.compute_risk_percentage(anomaly_scores, symbol)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'timestamp': df['timestamp'],
            'symbol': symbol,
            'risk_percentage': risk_percentages,
            'anomaly_score': anomaly_scores
        })
        
        return result
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Convert score_window to a regular dict for serialization
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'window_size': self.window_size,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'score_window': {k: v for k, v in self.score_window.items()}
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the state
        joblib.dump(state, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'IsolationForestModel':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded IsolationForestModel instance
        """
        try:
            state = joblib.load(filepath)
            
            # Create a new model instance
            model = cls(
                contamination=state['model'].contamination,
                n_estimators=len(state['model'].estimators_),
                random_state=state['model'].random_state,
                window_size=state['window_size']
            )
            
            # Restore the state
            model.model = state['model']
            model.scaler = state['scaler']
            model.feature_columns = state.get('feature_columns')
            if model.feature_columns is None:
                logger.warning("Model state missing feature_columns; will infer from input data at runtime")
            model.is_fitted = state['is_fitted']
            model.score_window = state.get('score_window', {})
            # Ensure score_window is a dict (it may have been serialized as list)
            if not isinstance(model.score_window, dict):
                logger.warning("score_window was not a dict; resetting")
                model.score_window = {}
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = IsolationForestModel(contamination=0.1, window_size=100)
    
    # Example: Train on some data
    # model.fit(X_train)
    
    # Example: Make predictions
    # results = model.predict(X_test, "BTC-USD")
    
    # Example: Save model
    # model.save("models/isolation_forest.pkl")
    
    # Example: Load model
    # loaded_model = IsolationForestModel.load("models/isolation_forest.pkl")