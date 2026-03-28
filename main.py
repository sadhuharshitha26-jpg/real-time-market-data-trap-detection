"""
MarketTrap - Real-time Market Anomaly Detection Pipeline

This script implements the complete data processing pipeline:
1. Streaming Ingestion → 2. Feature Engineering → 3. Anomaly Detection → 4. Risk Scoring → 5. Visualization
"""

import time
import logging
import webbrowser
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_trap.log')
    ]
)
logger = logging.getLogger(__name__)

# Import pipeline components
from data_ingestion.stream_ingest import MarketDataStream
from feature_engineering.spark_features import compute_features_spark
from ml_pipeline.anomaly_model import IsolationForestModel

class MarketTrapPipeline:
    """Main pipeline for real-time market trap detection."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to the pre-trained Isolation Forest model
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path or "models/isolation_forest.pkl"
        self.plot_path = Path("outputs/live_plot.html")
        self.plot_opened = False
        self.setup_pipeline()
    
    def setup_pipeline(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing MarketTrap pipeline...")
        self.load_models()
        logger.info("Pipeline initialization complete")
    
    def load_models(self) -> None:
        """Load the pre-trained Isolation Forest model and scaler."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = IsolationForestModel.load(self.model_path)
            logger.info("Successfully loaded model and scaler")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def process_stream(self, symbol: str = 'BTC-USD', batch_size: int = 10, 
                      interval: float = 1.0, max_messages: int = None) -> None:
        """
        Process the market data stream through the complete pipeline.
        
        Args:
            symbol: Trading pair symbol
            batch_size: Number of messages per batch
            interval: Time in seconds between batches
            max_messages: Maximum number of messages to process
        """
        logger.info(f"Starting pipeline for {symbol}...")
        
        # Initialize data stream
        stream = MarketDataStream(
            symbols=[symbol],
            batch_size=batch_size,
            interval=interval,
            max_messages=max_messages
        )
        
        try:
            for batch in stream.stream():
                for batch_symbol, messages in batch.items():
                    if not messages:
                        continue
                    logger.info(f"Processing batch of {len(messages)} messages for {batch_symbol}")

                    # 1. Convert batch to DataFrame
                    df = pd.DataFrame(messages)

                    # 2. Feature Engineering
                    logger.debug("Performing feature engineering...")
                    features_df = self._compute_features(df)

                    # 3. Anomaly Detection & Risk Scoring
                    logger.debug("Running anomaly detection...")
                    results = self._score_risk(features_df)

                    # 4. Visualization
                    self._visualize_results(results)

                    logger.info(
                        f"Processed batch. Current risk score: {results['risk_percentage'][-1]:.1f}%"
                    )
                
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        except Exception as e:
            logger.error(f"Error in processing pipeline: {str(e)}")
            raise
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for anomaly detection."""
        features = df.copy()
        
        # Ensure we have numeric types
        features['price'] = pd.to_numeric(features['price'])
        features['volume'] = pd.to_numeric(features['volume'])
        
        # 1. Price returns
        features['price_return'] = features['price'].pct_change()
        
        # 2. Volume changes
        features['volume_change'] = features['volume'].pct_change()
        
        # 3. Volatility (10-period rolling std of returns)
        features['volatility'] = features['price_return'].rolling(window=10).std()
        
        # 4. Breakout strength (price relative to 20-period high)
        window_size = min(20, len(features))
        features['rolling_max'] = features['price'].rolling(window=window_size, min_periods=1).max()
        features['breakout_strength'] = (features['price'] - features['rolling_max']) / features['rolling_max']
        features['is_breakout'] = (features['price'] > features['rolling_max'].shift(1)).astype(int)
        
        # 5. Price-Volume divergence
        # Positive return with decreasing volume, or negative return with increasing volume
        pv_corr = features['price_return'].rolling(window=10).corr(features['volume_change'])
        features['pv_divergence'] = ((features['price_return'] > 0) & (features['volume_change'] < 0)) | \
                                  ((features['price_return'] < 0) & (features['volume_change'] > 0))
        features['pv_divergence'] = features['pv_divergence'].astype(int)
        
        # Fill any remaining NA values
        features = features.fillna(0)
        
        # Ensure we have all required columns
        required_columns = [
            'price_return', 'volume_change', 'volatility',
            'breakout_strength', 'is_breakout', 'pv_divergence'
        ]
        
        # Add missing columns with default values if needed
        for col in required_columns:
            if col not in features.columns:
                features[col] = 0
        
        return features[required_columns + ['timestamp', 'price', 'volume']]
    
    def _score_risk(self, df: pd.DataFrame) -> Dict[str, List]:
        """Score the data using the Isolation Forest model."""
        try:
            # Get anomaly scores
            scores = self.model.anomaly_score(df)
            
            # Convert to risk percentages
            risk_pct = self.model.risk_percentage(scores)
            
            # Create results dictionary
            results = {
                'timestamp': df['timestamp'].tolist(),
                'price': df['price'].tolist(),
                'volume': df['volume'].tolist(),
                'anomaly_score': scores.tolist(),
                'risk_percentage': risk_pct.tolist() if isinstance(risk_pct, np.ndarray) else [risk_pct]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in risk scoring: {str(e)}")
            raise
    
    def _visualize_results(self, results: Dict[str, List]) -> None:
        """Visualize the pipeline results using Plotly with two subplots."""
        try:
            # Calculate baseline and deviation for risk percentage
            risk_pct = np.array(results['risk_percentage'])
            window_size = min(20, len(risk_pct) // 2 or 1)  # Dynamic window size
            
            # Calculate moving average and standard deviation
            if len(risk_pct) > window_size:
                rolling_mean = pd.Series(risk_pct).rolling(window=window_size, center=True, min_periods=1).mean().values
                rolling_std = pd.Series(risk_pct).rolling(window=window_size, center=True, min_periods=1).std().values
            else:
                rolling_mean = np.full_like(risk_pct, np.mean(risk_pct))
                rolling_std = np.zeros_like(risk_pct)
            
            # Find significant deviations (1.5 std dev from mean)
            deviation = np.abs(risk_pct - rolling_mean)
            significant = deviation > 1.5 * rolling_std
            significant_indices = np.where(significant)[0]
            
            # Create figure with two subplots
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Price and Volume", "Trap Risk Percentage"),
                row_heights=[0.6, 0.4]  # 60% for price/volume, 40% for risk
            )
            
            # Add price trace to first subplot
            fig.add_trace(
                go.Scatter(
                    x=results['timestamp'],
                    y=results['price'],
                    name="Price (USD)",
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='%{y:.2f} USD<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add volume trace to first subplot (secondary y-axis)
            fig.add_trace(
                go.Bar(
                    x=results['timestamp'],
                    y=results['volume'],
                    name="Volume",
                    opacity=0.3,
                    marker_color='#7f7f7f',
                    yaxis="y2",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add risk percentage trace to second subplot
            fig.add_trace(
                go.Scatter(
                    x=results['timestamp'],
                    y=results['risk_percentage'],
                    name="Trap Risk %",
                    line=dict(color='#ff7f0e', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 165, 0, 0.2)',
                    hovertemplate='%{y:.1f}%<extra></extra>',
                    mode='lines'  # Start with just lines
                ),
                row=2, col=1
            )
            
            # Add significant deviation markers
            if len(significant_indices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[results['timestamp'][i] for i in significant_indices],
                        y=[results['risk_percentage'][i] for i in significant_indices],
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='diamond',
                            line=dict(width=1, color='white')
                        ),
                        name='Significant Deviation',
                        hovertemplate='%{y:.1f}%<extra></extra>',
                        showlegend=True
                    ),
                    row=2, col=1
                )
            
            # Add baseline (moving average) line
            fig.add_trace(
                go.Scatter(
                    x=results['timestamp'],
                    y=rolling_mean,
                    line=dict(color='#2ca02c', width=1.5, dash='dot'),
                    name='Risk Baseline',
                    hovertemplate='Baseline: %{y:.1f}%<extra></extra>',
                    showlegend=True
                ),
                row=2, col=1
            )
            
            # Add confidence band
            fig.add_trace(
                go.Scatter(
                    x=results['timestamp'] + results['timestamp'][::-1],  # x then reversed x
                    y=np.concatenate([rolling_mean + 1.5*rolling_std, 
                                     (rolling_mean - 1.5*rolling_std)[::-1]]),  # upper then lower reversed
                    fill='toself',
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
            
            # Update y-axes
            fig.update_yaxes(
                title_text="Price (USD)",
                row=1, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey'
            )
            
            fig.update_yaxes(
                title_text="Volume",
                row=1, col=1,
                secondary_y=True,
                showgrid=False,
                showticklabels=True
            )
            
            fig.update_yaxes(
                title_text="Risk %",
                row=2, col=1,
                range=[0, 100],
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey'
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': "Market Trap Risk Analysis",
                    'y':0.98,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Time",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                height=800,
                hovermode='x unified',
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey'
                ),
                xaxis2=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey'
                )
            )
            
            # Write to a single HTML file and open once to avoid multiple tabs
            self.plot_path.parent.mkdir(parents=True, exist_ok=True)
            html_body = fig.to_html(include_plotlyjs="cdn", full_html=False)
            html = (
                "<!doctype html>"
                "<html><head>"
                "<meta charset='utf-8'>"
                "<meta http-equiv='refresh' content='2'>"
                "<title>MarketTrap Live</title>"
                "</head><body>"
                f"{html_body}"
                "</body></html>"
            )
            self.plot_path.write_text(html, encoding="utf-8")
            if not self.plot_opened:
                webbrowser.open(self.plot_path.resolve().as_uri(), new=0, autoraise=True)
                self.plot_opened = True
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            raise
    
def main():
    """Main entry point for the MarketTrap application."""
    try:
        logger.info("=== Starting MarketTrap Application ===")
        
        # Initialize and run the pipeline
        pipeline = MarketTrapPipeline()
        pipeline.process_stream(
            symbol='BTC-USD',
            batch_size=5,
            interval=2.0,
            max_messages=20
        )
        
    except Exception as e:
        logger.critical(f"Fatal error in MarketTrap: {str(e)}", exc_info=True)
    finally:
        logger.info("=== MarketTrap Application Stopped ===")

if __name__ == "__main__":
    main()