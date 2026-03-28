"""
MarketTrap Feature Engineering with PySpark

This module implements scalable feature engineering for market data using PySpark.
It processes multiple cryptocurrencies in parallel, with all window operations
properly partitioned by symbol.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pathlib import Path

def compute_features_spark(input_file=None, output_file=None):
    """
    Compute technical features for market data using PySpark.
    
    This function processes raw market data and generates numerical features
    that can be used for anomaly detection. Features are calculated separately
    for each cryptocurrency symbol.
    
    Args:
        input_file (str, optional): Path to the input CSV file. 
            Defaults to 'data/market_data.csv'.
        output_file (str, optional): Path to save the output CSV file. 
            Defaults to 'data/features_spark.csv'.
    """
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("MarketTrapFeatureEngineering") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
        
        # Set default paths
        project_root = Path(__file__).parent.parent
        input_file = input_file or str(project_root / 'data' / 'market_data.csv')
        output_file = output_file or str(project_root / 'data' / 'features_spark.csv')
        
        # Read the data
        print(f"Reading data from {input_file}...")
        df = spark.read.csv(input_file, header=True, inferSchema=True)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Input file is missing required column: {col}")
        
        # Ensure proper data types and sort
        df = (df
              .withColumn("timestamp", F.to_timestamp("timestamp"))
              .withColumn("close", F.col("close").cast("double"))
              .withColumn("volume", F.col("volume").cast("double"))
              .orderBy("symbol", "timestamp"))
        
        # Define window specifications - partitioned by symbol and ordered by timestamp
        window_spec = Window.partitionBy("symbol").orderBy("timestamp")
        window_20 = (Window.partitionBy("symbol")
                     .orderBy("timestamp")
                     .rowsBetween(-19, 0))
        
        # 1. Price change (percentage)
        df = df.withColumn(
            "price_return",
            (F.col("close") - F.lag("close", 1).over(window_spec)) / 
            F.lag("close", 1).over(window_spec)
        )
        
        # 2. Volume change (percentage)
        df = df.withColumn(
            "volume_change",
            (F.col("volume") - F.lag("volume", 1).over(window_spec)) / 
            F.when(F.lag("volume", 1).over(window_spec) > 0, 
                  F.lag("volume", 1).over(window_spec))
            .otherwise(1.0)  # Avoid division by zero
        )
        
        # 3. Volatility (20-day rolling standard deviation of returns)
        df = df.withColumn(
            "volatility",
            F.stddev("price_return").over(window_20)
        )
        
        # 4. Breakout strength (continuous measure)
        rolling_high = F.max("high").over(window_20)
        df = df.withColumn(
            "breakout_strength",
            (F.col("close") - rolling_high) / rolling_high
        )
        
        # 5. Price-volume correlation (20-day rolling)
        df = df.withColumn(
            "pv_correlation",
            F.corr("price_return", "volume_change").over(window_20)
        )
        
        # 6. Price momentum (5-period rate of change)
        window_5 = (Window.partitionBy("symbol")
                   .orderBy("timestamp")
                   .rowsBetween(-4, 0))
        df = df.withColumn(
            "price_momentum",
            (F.col("close") - F.lag("close", 4).over(window_spec)) / 
            F.lag("close", 4).over(window_spec)
        )
        
        # 7. Volume momentum (5-period rate of change)
        df = df.withColumn(
            "volume_momentum",
            (F.col("volume") - F.lag("volume", 4).over(window_spec)) / 
            F.when(F.lag("volume", 4).over(window_spec) > 0,
                  F.lag("volume", 4).over(window_spec))
            .otherwise(1.0)
        )
        
        # Fill null values that occur at the beginning of each symbol's data
        numeric_cols = [col for col in df.columns if col not in ['timestamp', 'symbol']]
        for col in numeric_cols:
            df = df.withColumn(col, F.when(F.col(col).isNull(), 0).otherwise(F.col(col)))
        
        # Select and reorder columns
        result_cols = [
            "timestamp", "symbol", "open", "high", "low", "close", "volume",
            "price_return", "volume_change", "volatility",
            "breakout_strength", "pv_correlation",
            "price_momentum", "volume_momentum"
        ]
        
        # Ensure all columns exist (in case input data has different structure)
        result_cols = [col for col in result_cols if col in df.columns]
        result_df = df.select(*result_cols)
        
        # Save the results
        print(f"Saving features to {output_file}...")
        (result_df
         .repartition("symbol")  # Better partitioning for output
         .write
         .partitionBy("symbol")  # Create partitioned output by symbol
         .mode('overwrite')
         .option("header", "true")
         .csv(output_file.replace('.csv', '')))
        
        print("Feature engineering completed successfully!")
        return result_df
        
    except Exception as e:
        print(f"Error in compute_features_spark: {str(e)}")
        raise

if __name__ == "__main__":
    compute_features_spark()