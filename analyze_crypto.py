from src.visualization.crypto_visualizer import CryptoVisualizer
from src.visualization.cluster_visualizer import ClusterVisualizer
from src.data_processing.data_merger import CryptoDataMerger
from src.clustering.clustering_methods import CryptoClusterer
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

class CryptoAnalyzer:
    def __init__(self):
        self.crypto_viz = CryptoVisualizer()
        self.cluster_viz = ClusterVisualizer()
        self.merger = CryptoDataMerger()
        self.clusterer = CryptoClusterer()
    
    def load_and_prepare_data(self, days=30):
        """Load and prepare recent cryptocurrency data"""
        print("Loading cryptocurrency data...")
        merged_df = self.merger.merge_crypto_data()
        
        # Filter out empty dataframes
        valid_symbols = merged_df.groupby(level=0).size()
        valid_symbols = valid_symbols[valid_symbols > 0].index
        merged_df = merged_df.loc[valid_symbols]
        
        # Get recent data
        end_date = merged_df.index.get_level_values('timestamp').max()
        start_date = end_date - timedelta(days=days)
        recent_data = self.merger.get_date_range_data(merged_df, start_date, end_date)
        
        if recent_data.empty:
            raise ValueError("No recent data available for analysis")
            
        return recent_data
    
    def perform_clustering(self, data):
        """Perform clustering on prepared data"""
        print("Preparing features...")
        features = self.clusterer.prepare_features(data)
        
        if features.empty:
            raise ValueError("No features could be calculated")
        
        print("Finding optimal number of clusters...")
        optimal_k, silhouette_scores = self.clusterer.get_optimal_clusters(features)
        print(f"Optimal number of clusters: {optimal_k}")
        
        print("Clustering cryptocurrencies...")
        labels, silhouette, centers = self.clusterer.kmeans_clustering(features, optimal_k)
        
        clusters_df = pd.DataFrame({
            'cluster': labels,
            'silhouette': silhouette
        }, index=features.index)
        
        return clusters_df, centers
    
    def calculate_cluster_statistics(self, cluster_data):
        """Calculate key statistics for a cluster"""
        stats = {
            'avg_7d_return': cluster_data['close'].pct_change(7).mean() * 100,
            'avg_volatility': cluster_data['close'].pct_change().std() * np.sqrt(252) * 100,
            'avg_volume': cluster_data['volume'].mean(),
            'price_correlation': cluster_data.groupby(level=0)['close'].mean().corr(cluster_data.groupby(level=0)['volume'].mean())
        }
        return stats
    
    def visualize_cluster(self, cluster_id, cluster_symbols, recent_data, clusters_df):
        """Visualize all cryptocurrencies in a cluster"""
        print(f"\nVisualizing Cluster {cluster_id}...")
        
        # Create cluster overview visualization
        self.cluster_viz.plot_cluster_comparison(
            recent_data.loc[cluster_symbols],
            clusters_df.loc[cluster_symbols],
            f"Cluster_{cluster_id}"
        )
        
        # Visualize each cryptocurrency in the cluster
        for symbol in cluster_symbols:
            try:
                print(f"Visualizing {symbol}...")
                
                # Price and indicator analysis
                self.crypto_viz.analyze_single_crypto(
                    symbol=symbol,
                    days=30,
                    include_indicators=True
                )
                
                # Cluster analysis visualization
                symbol_data = self.merger.get_symbol_data(recent_data, symbol)
                symbol_clusters = clusters_df[clusters_df.index.get_level_values('symbol') == symbol]
                
                self.cluster_viz.plot_cluster_analysis(
                    df=symbol_data,
                    clusters_df=symbol_clusters,
                    symbol=symbol,
                    method='kmeans'
                )
                
            except Exception as e:
                print(f"Error visualizing {symbol}: {e}")
    
    def analyze_clusters(self, clusters_df, recent_data):
        """Analyze and visualize all clusters"""
        for cluster_id in clusters_df['cluster'].unique():
            # Get symbols for this cluster
            cluster_symbols = clusters_df[clusters_df['cluster'] == cluster_id].index.get_level_values('symbol').unique()
            if len(cluster_symbols) == 0:
                continue
            
            print(f"\nAnalyzing Behavior Cluster {cluster_id}:")
            print(f"Number of cryptocurrencies: {len(cluster_symbols)}")
            print("Cryptocurrencies:", ', '.join(cluster_symbols))
            
            # Calculate and print cluster statistics
            cluster_data = recent_data[recent_data.index.get_level_values('symbol').isin(cluster_symbols)]
            stats = self.calculate_cluster_statistics(cluster_data)
            
            print("\nCluster Statistics:")
            print(f"Average 7-day return: {stats['avg_7d_return']:.2f}%")
            print(f"Average annualized volatility: {stats['avg_volatility']:.2f}%")
            print(f"Average volume: {stats['avg_volume']:.2f}")
            print(f"Price-Volume correlation: {stats['price_correlation']:.2f}")
            
            # Visualize the cluster
            self.visualize_cluster(cluster_id, cluster_symbols, recent_data, clusters_df)
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # Load and prepare data
            recent_data = self.load_and_prepare_data(days=30)
            
            # Perform clustering
            clusters_df, centers = self.perform_clustering(recent_data)
            
            # Analyze and visualize clusters
            self.analyze_clusters(clusters_df, recent_data)
            
            print("\nAnalysis complete. Visualizations have been created in the data/plots directory.")
            
        except Exception as e:
            print(f"Error during analysis: {e}")

def main():
    analyzer = CryptoAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 