from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.clustering.regime_detection.short_term import ShortTermDetector
from src.clustering.methods.clustering import KMeansClusterer, GraphClusterer
from src.visualization.regime_visualizer import RegimeVisualizer
from src.data_processing.data_merger import CryptoDataMerger

class RegimeAnalyzer:
    def __init__(self):
        self.data_dir = Path("data")
        self.plots_dir = self.data_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.merger = CryptoDataMerger()
        self.regime_detector = ShortTermDetector()
        self.kmeans_clusterer = KMeansClusterer(n_clusters=5)
        self.graph_clusterer = GraphClusterer(threshold=0.5)
        self.visualizer = RegimeVisualizer()
    
    def analyze_single_crypto(self, symbol: str, df: pd.DataFrame):
        """Analyze regimes for a single cryptocurrency"""
        print(f"\nAnalyzing {symbol}...")
        
        # Detect regimes
        regimes, confidence = self.regime_detector.detect_regime(df)
        
        # Analyze transitions
        transitions = self.regime_detector.analyze_regime_transitions(regimes)
        
        # Visualize results
        self.visualizer.plot_regime_analysis(
            df=df,
            regimes=regimes,
            symbol=symbol,
            save_path=self.plots_dir / f"{symbol.replace('/', '_')}_regime_analysis.html"
        )
        
        # Visualize transitions
        if not transitions.empty:
            self.visualizer.plot_regime_transitions(
                transitions,
                save_path=self.plots_dir / f"{symbol.replace('/', '_')}_transitions.html"
            )
            
            # Print transition statistics
            print(f"\nRegime Transition Statistics for {symbol}:")
            print(f"Average regime duration: {transitions.attrs['avg_regime_duration']:.2f} hours")
            print(f"Most common transition: {transitions.attrs['most_common_transition']}")
            print(f"Number of regime changes: {len(transitions)}")
        
        return regimes, confidence, transitions
    
    def cluster_cryptos(self, all_features: pd.DataFrame):
        """Cluster cryptocurrencies based on their features"""
        print("\nClustering cryptocurrencies...")
        
        try:
            if all_features.empty:
                print("No features available for clustering!")
                return pd.DataFrame(), {}
            
            if len(all_features) < 2:
                print("Not enough samples for clustering!")
                return pd.DataFrame(), {}
            
            # Try both clustering methods
            try:
                kmeans_clusters, kmeans_metrics = self.kmeans_clusterer.fit(all_features)
                kmeans_valid = (
                    kmeans_metrics.get('silhouette_score', -1) > 0 and 
                    not kmeans_clusters.empty
                )
            except Exception as e:
                print(f"K-means clustering failed: {str(e)}")
                kmeans_valid = False
            
            try:
                graph_clusters, graph_metrics = self.graph_clusterer.fit(all_features)
                graph_valid = (
                    graph_metrics.get('modularity', -1) > 0 and 
                    not graph_clusters.empty
                )
            except Exception as e:
                print(f"Graph clustering failed: {str(e)}")
                graph_valid = False
            
            # Select the better method
            if kmeans_valid and (not graph_valid or 
                               kmeans_metrics['silhouette_score'] > graph_metrics['modularity']):
                print("Using K-means clustering...")
                return kmeans_clusters, kmeans_metrics
            elif graph_valid:
                print("Using graph-based clustering...")
                return graph_clusters, graph_metrics
            else:
                print("No valid clustering results!")
                return pd.DataFrame(), {}
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            return pd.DataFrame(), {}
    
    def analyze_clusters(self, clusters: pd.DataFrame, dfs: dict):
        """Analyze and visualize clusters"""
        print("\nAnalyzing clusters...")
        
        try:
            unique_clusters = sorted(clusters['cluster'].unique())
            if len(unique_clusters) == 0:
                print("No clusters found!")
                return
            
            for cluster_id in unique_clusters:
                try:
                    # Get symbols in this cluster
                    cluster_mask = clusters['cluster'] == cluster_id
                    if not cluster_mask.any():
                        continue
                    
                    cluster_symbols = clusters[cluster_mask].index.unique()
                    if len(cluster_symbols) == 0:
                        continue
                    
                    # Get data for these symbols
                    cluster_dfs = {}
                    cluster_labels = {}
                    
                    for symbol in cluster_symbols:
                        if symbol in dfs:
                            cluster_dfs[symbol] = dfs[symbol]
                            cluster_labels[symbol] = f"Cluster {cluster_id}"
                    
                    if not cluster_dfs:
                        continue
                    
                    # Visualize cluster
                    self.visualizer.plot_cluster_comparison(
                        dfs=cluster_dfs,
                        cluster_labels=cluster_labels,
                        save_path=self.plots_dir / f"cluster_{cluster_id}_comparison.html"
                    )
                    
                    # Calculate cluster statistics
                    returns = []
                    volatilities = []
                    volumes = []
                    
                    for df in cluster_dfs.values():
                        returns.append(df['close'].pct_change().mean() * 100)
                        volatilities.append(df['close'].pct_change().std() * np.sqrt(252) * 100)
                        volumes.append(df['volume'].mean())
                    
                    print(f"\nCluster {cluster_id}:")
                    print(f"Number of cryptocurrencies: {len(cluster_symbols)}")
                    print("Symbols:", ", ".join(cluster_symbols))
                    print(f"Average daily return: {np.mean(returns):.2f}%")
                    print(f"Average annualized volatility: {np.mean(volatilities):.2f}%")
                    print(f"Average daily volume: {np.mean(volumes):.2f}")
                    
                except Exception as e:
                    print(f"Error analyzing cluster {cluster_id}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Error in cluster analysis: {str(e)}")
    
    def run_analysis(self, days: int = 30):
        """Run complete regime analysis"""
        try:
            # Load data
            print("Loading cryptocurrency data...")
            merged_df = self.merger.merge_crypto_data()
            
            if merged_df.empty:
                print("No data available for analysis!")
                return
            
            # Filter recent data
            end_date = merged_df.index.get_level_values('timestamp').max()
            start_date = end_date - timedelta(days=days)
            recent_data = merged_df[merged_df.index.get_level_values('timestamp') >= start_date]
            
            if recent_data.empty:
                print("No recent data available!")
                return
            
            # Analyze each cryptocurrency
            all_regimes = {}
            all_features = pd.DataFrame()
            symbol_dfs = {}
            
            symbols = recent_data.index.get_level_values('symbol').unique()
            if len(symbols) == 0:
                print("No symbols found in the data!")
                return
            
            for symbol in symbols:
                try:
                    df = recent_data.xs(symbol, level='symbol')
                    if df.empty:
                        continue
                    
                    symbol_dfs[symbol] = df
                    
                    # Detect regimes
                    regimes, confidence, transitions = self.analyze_single_crypto(symbol, df)
                    all_regimes[symbol] = regimes
                    
                    # Prepare features for clustering
                    features = self.regime_detector.prepare_features(df)
                    if not features.empty:
                        all_features = pd.concat([all_features, features.mean().to_frame().T])
                    
                except Exception as e:
                    print(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            if all_features.empty:
                print("No features available for clustering!")
                return
            
            # Cluster cryptocurrencies
            clusters, metrics = self.cluster_cryptos(all_features)
            
            if not clusters.empty:
                # Analyze clusters
                self.analyze_clusters(clusters, symbol_dfs)
            
            print("\nAnalysis complete! Check the plots directory for visualizations.")
            
        except Exception as e:
            print(f"Error during analysis: {e}")

def main():
    analyzer = RegimeAnalyzer()
    analyzer.run_analysis(days=30)

if __name__ == "__main__":
    main() 