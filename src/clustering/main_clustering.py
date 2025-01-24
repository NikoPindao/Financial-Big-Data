from src.clustering.clustering_methods import CryptoClusterer
from src.visualization.cluster_visualizer import ClusterVisualizer
from src.data_processing.data_fetching import CryptoDataFetcher
import pandas as pd

def main(symbol="BTC/USDT"):
    """
    Run clustering analysis for a single cryptocurrency.
    
    Args:
        symbol (str): Trading pair symbol (e.g., "BTC/USDT")
    """
    # Initialize classes
    clusterer = CryptoClusterer()
    visualizer = ClusterVisualizer()
    fetcher = CryptoDataFetcher()
    
    # Load data
    print(f"Loading data for {symbol}...")
    df = fetcher.load_symbol_data(symbol)
    
    # Prepare features
    print("Preparing features...")
    features = clusterer.prepare_features(df)
    
    # Apply different clustering methods
    print("Applying clustering methods...")
    
    # 1. Louvain
    print("Running Louvain clustering...")
    communities, modularity, G = clusterer.louvain_clustering(features)
    
    # 2. DBSCAN
    print("Running DBSCAN clustering...")
    dbscan_labels, dbscan_silhouette = clusterer.dbscan_clustering(features)
    
    # 3. K-means with optimal clusters
    print("Finding optimal number of clusters...")
    optimal_k, silhouette_scores = clusterer.get_optimal_clusters(features)
    print(f"Optimal number of clusters: {optimal_k}")
    
    print("Running K-means clustering...")
    kmeans_labels, kmeans_silhouette, centers = clusterer.kmeans_clustering(features, optimal_k)
    
    # Create DataFrames with clustering results
    clusters_dict = {
        'Louvain': pd.DataFrame({
            'cluster': communities,
            'modularity': modularity
        }, index=features.index),
        'DBSCAN': pd.DataFrame({
            'cluster': dbscan_labels,
            'silhouette': dbscan_silhouette
        }, index=features.index),
        'KMeans': pd.DataFrame({
            'cluster': kmeans_labels,
            'silhouette': kmeans_silhouette
        }, index=features.index)
    }
    
    # Visualize results
    print("Creating visualizations...")
    visualizer.plot_cluster_comparison(df, clusters_dict, symbol)
    visualizer.plot_optimal_clusters(optimal_k, silhouette_scores)
    
    print(f"Clustering analysis complete for {symbol}")

if __name__ == "__main__":
    main() 