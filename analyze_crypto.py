from src.visualization.crypto_visualizer import CryptoVisualizer
from src.data_processing.data_fetching import CryptoDataFetcher
import pandas as pd

def analyze_bitcoin():
    # Initialize visualizer and data fetcher
    visualizer = CryptoVisualizer()
    fetcher = CryptoDataFetcher()
    
    # Load BTC data
    symbol = "BTC/USDT"
    print(f"Loading {symbol} data...")
    df = fetcher.load_symbol_data(symbol)
    
    # Create basic analysis visualization
    print("Creating price and technical analysis visualization...")
    visualizer.analyze_single_crypto(
        symbol=symbol,
        days=365,  # Show last year
        include_indicators=True  # Include technical indicators
    )
    
    # Create cluster analysis
    print("Creating cluster analysis visualization...")
    # Create dummy clusters for demonstration
    clusters_df = pd.DataFrame({
        'cluster': [0, 0, 1, 1, 2, 2, 1, 1, 0, 0] * (len(df) // 10),
        'silhouette': [0.8] * len(df)
    }, index=df.index[:len(df) - len(df) % 10])
    
    visualizer.plot_cluster_analysis(
        df=df,
        clusters_df=clusters_df,
        symbol=symbol,
        method='example'
    )
    
    print("\nVisualizations have been created in the data/plots directory:")
    print("1. BTC_USDT_analysis.html - Price and technical indicators")
    print("2. BTC_USDT_example_clusters.html - Cluster analysis")

if __name__ == "__main__":
    analyze_bitcoin() 