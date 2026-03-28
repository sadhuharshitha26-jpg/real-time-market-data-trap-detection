"""
MarketTrap - Main Execution Script

This script runs the complete MarketTrap pipeline:
1. Data Ingestion
2. Feature Engineering
3. Anomaly Detection
4. Risk Scoring
5. Visualization
"""

import os
import sys
import logging
from pathlib import Path

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

def check_environment():
    """Check if all required environment variables and files exist."""
    logger.info("Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check required directories
    required_dirs = [
        'data',
        'models',
        'outputs',
        'data_ingestion',
        'feature_engineering',
        'ml_pipeline'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.error(f"Required directory not found: {dir_path}")
            return False
    
    # Check required files
    required_files = [
        'data/btc_usd.csv',
        'ml_pipeline/anomaly_model.py',
        'ml_pipeline/train_anomaly_model.py',
        'feature_engineering/spark_features.py',
        'data_ingestion/stream_ingest.py',
        'main.py'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            return False
    
    logger.info("Environment check passed")
    return True

def install_dependencies():
    """Install required Python packages."""
    logger.info("Installing dependencies...")
    try:
        import pip
        import importlib
        
        required_packages = [
            'pandas',
            'numpy',
            'scikit-learn',
            'matplotlib',
            'seaborn',
            'yfinance',
            'joblib',
            'pyspark',
            'mplfinance',
            'plotly',
            'kafka-python'  # For future Kafka integration
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                logger.debug(f"{package} is already installed")
            except ImportError:
                logger.info(f"Installing {package}...")
                pip.main(['install', package])
        
        logger.info("All dependencies are installed")
        return True
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def main():
    """Main entry point for the MarketTrap application."""
    try:
        logger.info("=== Starting MarketTrap ===")
        
        # Check environment
        if not check_environment():
            logger.error("Environment check failed. Please fix the issues and try again.")
            return 1
        
        # Install dependencies if needed
        if not install_dependencies():
            logger.error("Failed to install dependencies")
            return 1
        
        # Import main application
        from main import main as run_app
        
        # Run the application
        logger.info("Starting MarketTrap application...")
        run_app()
        
        logger.info("=== MarketTrap completed successfully ===")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nMarketTrap stopped by user")
        return 0
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
