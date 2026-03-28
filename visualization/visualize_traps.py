import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime

def plot_traps(scores_file=None, output_file=None):
    """
    Plot Bitcoin price and trap confidence scores over time.
    
    Args:
        scores_file (str, optional): Path to the trap scores CSV file. 
            Defaults to 'outputs/trap_scores.csv'.
        output_file (str, optional): Path to save the output figure.
            Defaults to 'reports/figs/trap_detection.png'.
    """
    try:
        # Set default paths
        project_root = Path(__file__).parent.parent
        scores_file = scores_file or str(project_root / 'outputs' / 'trap_scores.csv')
        output_file = output_file or str(project_root / 'reports' / 'figs' / 'trap_detection.png')
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load the data
        print(f"Loading trap scores from {scores_file}...")
        df = pd.read_csv(scores_file)
        
        # Convert date to datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Identify high-risk periods
        high_risk = df[df['risk_category'] == 'High Trap Risk']
        
        # Create figure and subplots
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Bitcoin Close Price
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(df['date'], df['close'], color='#0d6efd', linewidth=2, label='BTC Price')
        plt.title('Bitcoin Price with High Trap Risk Periods', fontsize=14, pad=20)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Highlight high-risk periods on price chart
        for _, row in high_risk.iterrows():
            plt.axvspan(
                row['date'] - pd.Timedelta(days=0.5),
                row['date'] + pd.Timedelta(days=0.5),
                color='red',
                alpha=0.3,
                label='_nolegend_'
            )
        
        # Add legend to first subplot only
        plt.legend(loc='upper left')
        
        # Plot 2: Trap Confidence Score
        plt.subplot(2, 1, 2, sharex=ax1)
        plt.bar(df['date'], df['trap_score'], 
               color=df['risk_category'].map({
                   'Normal': '#20c997',
                   'Risky': '#ffc107',
                   'High Trap Risk': '#dc3545'
               }),
               width=1.0, alpha=0.7)
        
        plt.title('Trap Confidence Score', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.ylim(0, 105)  # Add some padding at the top
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Successfully saved visualization to {output_file}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    plot_traps()