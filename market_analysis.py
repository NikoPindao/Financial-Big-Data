from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from community import community_louvain
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import traceback
from sklearn.metrics import silhouette_score

from src.clustering.regime_detection.short_term import ShortTermDetector
from src.clustering.methods.clustering import KMeansClusterer, GraphClusterer
from src.visualization.regime_visualizer import RegimeVisualizer
from src.data_processing.data_merger import CryptoDataMerger
from src.features.feature_engineering import FeatureEngineer
from src.clustering.cluster_utils import ClusteringUtils
from src.models.lstm_regime_predictor import predict_regimes

class MarketAnalyzer:
    def __init__(self):
        self.data_dir = Path("data")
        self.plots_dir = self.data_dir / "plots/market_analysis"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.merger = CryptoDataMerger()
        self.regime_detector = ShortTermDetector()
        self.visualizer = RegimeVisualizer()
    
    def analyze_market_regimes(self, data: pd.DataFrame) -> Dict:
        """Analyze market-wide regime distribution"""
        all_regimes = {}
        regime_stats = {
            'stable_bull': 0,
            'volatile_bull': 0,
            'stable_bear': 0,
            'volatile_bear': 0,
            'consolidation': 0
        }
        
        # Detect regimes for each crypto
        for symbol in data.index.get_level_values('symbol').unique():
            df = data.xs(symbol, level='symbol')
            regimes, _ = self.regime_detector.detect_regime(df)
            all_regimes[symbol] = regimes
            
            # Count regime occurrences
            for regime in regime_stats.keys():
                regime_stats[regime] += (regimes['regime'] == regime).sum()
        
        return all_regimes, regime_stats
    
    def plot_market_regime_distribution(self, regime_stats: Dict, save_path: str):
        """Visualize market-wide regime distribution"""
        fig = go.Figure(data=[
            go.Pie(
                labels=list(regime_stats.keys()),
                values=list(regime_stats.values()),
                hole=.3
            )
        ])
        
        fig.update_layout(
            title="Market-wide Regime Distribution",
            annotations=[{
                'text': 'Regimes',
                'showarrow': False,
                'font_size': 20
            }]
        )
        
        fig.write_html(save_path)
        return fig
    
    def analyze_regime_correlations(self, all_regimes: Dict) -> pd.DataFrame:
        """Analyze correlations between different cryptocurrencies' regimes"""
        regime_matrix = pd.DataFrame()
        
        for symbol, regimes in all_regimes.items():
            # Convert regimes to numeric values
            regime_map = {
                'stable_bull': 1,
                'volatile_bull': 2,
                'stable_bear': -1,
                'volatile_bear': -2,
                'consolidation': 0
            }
            regime_matrix[symbol] = regimes['regime'].map(regime_map)
        
        return regime_matrix.corr()
    
    def plot_regime_correlations(self, corr_matrix: pd.DataFrame, save_path: str):
        """Visualize regime correlations between cryptocurrencies"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title="Regime Correlation Matrix",
            height=800,
            width=800
        )
        
        fig.write_html(save_path)
        return fig
    
    def analyze_market_transitions(self, all_regimes: Dict):
        """Analyze market-wide regime transitions"""
        all_transitions = []
        
        for symbol, regimes in all_regimes.items():
            try:
                transitions = self.regime_detector.analyze_regime_transitions(regimes)
                if not transitions.empty:
                    transitions['symbol'] = symbol
                    all_transitions.append(transitions)
            except Exception as e:
                print(f"Error analyzing transitions for {symbol}: {e}")
                continue
        
        if not all_transitions:
            # Return empty DataFrame with expected columns if no transitions
            return pd.DataFrame(columns=[
                'from_regime', 'to_regime', 'start_time', 'end_time',
                'duration', 'duration_days', 'symbol',
                'is_volatility_increase', 'is_trend_change'
            ])
        
        market_transitions = pd.concat(all_transitions, ignore_index=True)
        
        # Calculate additional metrics if needed
        if 'duration' in market_transitions.columns and 'duration_days' not in market_transitions.columns:
            market_transitions['duration_days'] = market_transitions['duration'] / 24
        
        return market_transitions
    
    def plot_market_transitions(self, transitions: pd.DataFrame, save_path: str):
        """Visualize market-wide transition patterns"""
        if transitions.empty:
            print("No transitions to visualize")
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Regime Duration by Type',
                'Most Common Transitions',
                'Transition Types',
                'Regime Stability'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        try:
            # Average duration by regime type
            duration_by_regime = transitions.groupby('from_regime')['duration_days'].mean()
            fig.add_trace(
                go.Bar(
                    x=duration_by_regime.index,
                    y=duration_by_regime.values,
                    name='Avg Duration'
                ),
                row=1, col=1
            )
            
            # Most common transitions
            transition_counts = transitions.groupby(['from_regime', 'to_regime']).size()
            top_transitions = transition_counts.nlargest(5)
            fig.add_trace(
                go.Bar(
                    x=[f"{f}->{t}" for f, t in top_transitions.index],
                    y=top_transitions.values,
                    name='Common Transitions'
                ),
                row=1, col=2
            )
            
            # Transition types
            if 'is_volatility_increase' in transitions.columns and 'is_trend_change' in transitions.columns:
                type_dist = pd.DataFrame({
                    'Type': ['Volatility Increase', 'Trend Change'],
                    'Count': [
                        transitions['is_volatility_increase'].sum(),
                        transitions['is_trend_change'].sum()
                    ]
                })
                fig.add_trace(
                    go.Bar(
                        x=type_dist['Type'],
                        y=type_dist['Count'],
                        name='Transition Types'
                    ),
                    row=2, col=1
                )
            
            # Regime stability
            stability = transitions.groupby('symbol')['duration_days'].mean()
            fig.add_trace(
                go.Box(
                    y=stability,
                    name='Regime Stability'
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=1000, title_text="Market-wide Transition Analysis")
            fig.write_html(save_path)
            return fig
        
        except Exception as e:
            print(f"Error creating transition visualization: {e}")
            return None
    
    def analyze_market_clusters(self, data: pd.DataFrame, period_name: str) -> Dict:
        """Analyze market clusters using multiple methods"""
        print("Preparing features for clustering...")
        features = pd.DataFrame()
        
        # Filter out problematic symbols
        excluded_symbols = ['FDUSD/USDT', 'EUR/USDT', 'USDC/USDT']
        valid_symbols = [s for s in data.index.get_level_values('symbol').unique() 
                        if s not in excluded_symbols]
        
        # Prepare features for all cryptocurrencies
        for symbol in valid_symbols:
            try:
                df = data.xs(symbol, level='symbol')
                if df.empty:
                    continue
                
                # Calculate comprehensive features
                returns = df['close'].pct_change()
                volume_ma = df['volume'].rolling(window=24).mean()
                
                # Get regime information
                regimes, _ = self.regime_detector.detect_regime(df)
                transitions = self.regime_detector.analyze_regime_transitions(regimes)
                
                # Only proceed if we have valid regime transitions
                if not transitions.empty and 'duration_days' in transitions.columns:
                    avg_duration = transitions['duration_days'].mean()
                else:
                    avg_duration = 0
                
                symbol_features = pd.Series({
                    # Price-based features
                    'volatility': returns.std() * np.sqrt(252),
                    'daily_return': returns.mean() * 252,
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    
                    # Volume-based features
                    'volume_trend': (df['volume'] / volume_ma).mean(),
                    'volume_volatility': (df['volume'] / volume_ma).std(),
                    
                    # Regime-based features
                    'regime_changes': len(regimes['regime'].unique()),
                    'avg_regime_duration': avg_duration,
                    
                    # Momentum features
                    'momentum_1d': df['close'].pct_change(24).mean(),
                    'momentum_1w': df['close'].pct_change(168).mean(),
                    
                    # Additional features
                    'high_low_ratio': (df['high'] / df['low']).mean(),
                    'price_range': ((df['high'] - df['low']) / df['close']).mean(),
                    'volume_price_corr': df['volume'].corr(df['close'])
                })
                
                features = pd.concat([features, symbol_features.to_frame(symbol).T])
                
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        if features.empty:
            raise ValueError("No features could be calculated")
        
        # Fill any remaining NaN values with column means
        features = features.fillna(features.mean())
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
        # Apply different clustering methods
        results = {}
        
        # 1. K-means clustering
        print("Applying K-means clustering...")
        kmeans = KMeansClusterer(n_clusters=5, n_init=10)
        kmeans_clusters, kmeans_metrics = kmeans.fit(scaled_features)
        results['kmeans'] = {
            'clusters': kmeans_clusters,
            'metrics': kmeans_metrics
        }
        
        # 2. Graph-based clustering (Louvain)
        print("Applying Louvain clustering...")
        # Create correlation network
        corr_matrix = scaled_features.T.corr()
        G = nx.Graph()
        
        # Add edges with weights based on correlation
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                weight = abs(corr_matrix.iloc[i,j])
                if weight > 0.3:  # Increased threshold for stronger connections
                    G.add_edge(corr_matrix.index[i], corr_matrix.index[j], weight=weight)
        
        # Apply Louvain method
        communities = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(communities, G)
        
        results['louvain'] = {
            'clusters': pd.DataFrame({
                'cluster': pd.Series(communities),
                'modularity': modularity
            }, index=features.index),
            'metrics': {
                'modularity': modularity,
                'n_communities': len(set(communities.values())),
                'graph_density': nx.density(G)
            }
        }
        
        # 3. DBSCAN clustering
        print("Applying DBSCAN clustering...")
        try:
            # Calculate optimal eps using nearest neighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(scaled_features)
            distances = nn.kneighbors(scaled_features)[0]
            eps = np.percentile(distances[:, 1], 90)
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=3)
            labels = dbscan.fit_predict(scaled_features)
            
            results['dbscan'] = {
                'clusters': pd.DataFrame({
                    'cluster': labels,
                    'noise': labels == -1
                }, index=features.index),
                'metrics': {
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'noise_points': sum(labels == -1),
                    'eps_used': eps
                }
            }
        except Exception as e:
            print(f"Error in DBSCAN clustering: {str(e)}")
            results['dbscan'] = {
                'clusters': pd.DataFrame({
                    'cluster': np.zeros(len(features)),
                    'noise': np.zeros(len(features), dtype=bool)
                }, index=features.index),
                'metrics': {
                    'n_clusters': 0,
                    'noise_points': 0,
                    'eps_used': 0.0
                }
            }
        
        return results, features
    
    def plot_louvain_network(self, G: nx.Graph, communities: Dict, features: pd.DataFrame, save_path: str):
        """Create interactive network visualization of Louvain communities"""
        # Calculate node positions using Force Atlas 2
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        
        # Prepare node traces for each community
        node_traces = []
        community_colors = px.colors.qualitative.Set3
        
        for community_id in set(communities.values()):
            # Get nodes in this community
            community_nodes = [node for node, com in communities.items() if com == community_id]
            
            # Create node trace
            node_x = [pos[node][0] for node in community_nodes]
            node_y = [pos[node][1] for node in community_nodes]
            
            # Get node statistics
            node_stats = features.loc[community_nodes]
            node_text = [
                f"Symbol: {node}<br>"
                f"Volatility: {features.loc[node, 'volatility']:.2%}<br>"
                f"Daily Return: {features.loc[node, 'daily_return']:.2%}<br>"
                f"Volume Trend: {features.loc[node, 'volume_trend']:.2f}"
                for node in community_nodes
            ]
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                name=f'Community {community_id}',
                text=community_nodes,
                hovertext=node_text,
                marker=dict(
                    size=15,
                    color=community_colors[community_id % len(community_colors)],
                    line=dict(width=1, color='white')
                )
            )
            node_traces.append(node_trace)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for (u, v, d) in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(d['weight'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace] + node_traces)
        
        fig.update_layout(
            title='Cryptocurrency Market Structure (Louvain Communities)',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            width=1000
        )
        
        fig.write_html(save_path)
        return fig
    
    def plot_enhanced_clustering_comparison(self, cluster_results: Dict, features: pd.DataFrame, save_path: str):
        """Create enhanced clustering comparison visualization"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'K-means Clusters', 'Louvain Communities', 'DBSCAN Clusters',
                'Feature Distribution (K-means)', 'Community Characteristics', 'DBSCAN Groups',
                'Cluster Stability', 'Community Network', 'Noise Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "heatmap"}, {"type": "violin"}],
                [{"type": "bar"}, {"type": "scatter3d"}, {"type": "bar"}]
            ]
        )
        
        # Plot clusters (first row)
        methods = ['kmeans', 'louvain', 'dbscan']
        for i, method in enumerate(methods):
            clusters = cluster_results[method]['clusters']
            
            # Create scatter plot
            for cluster in sorted(clusters['cluster'].unique()):
                mask = clusters['cluster'] == cluster
                cluster_features = features[mask]
                
                fig.add_trace(
                    go.Scatter(
                        x=cluster_features['volatility'],
                        y=cluster_features['daily_return'],
                        mode='markers',
                        name=f'{method.capitalize()} {cluster}',
                        text=cluster_features.index,
                        marker=dict(size=10)
                    ),
                    row=1, col=i+1
                )
        
        # Second row - detailed analysis
        # K-means feature distribution
        kmeans_clusters = cluster_results['kmeans']['clusters']
        fig.add_trace(
            go.Box(
                x=kmeans_clusters['cluster'],
                y=features['volatility'],
                name='Volatility'
            ),
            row=2, col=1
        )
        
        # Louvain community characteristics
        community_features = features.groupby(cluster_results['louvain']['clusters']['cluster']).mean()
        fig.add_trace(
            go.Heatmap(
                z=community_features.values,
                x=community_features.columns,
                y=[f'Community {i}' for i in range(len(community_features))],
                colorscale='Viridis'
            ),
            row=2, col=2
        )
        
        # DBSCAN distribution
        dbscan_clusters = cluster_results['dbscan']['clusters']
        fig.add_trace(
            go.Violin(
                x=dbscan_clusters['cluster'],
                y=features['daily_return'],
                name='Returns'
            ),
            row=2, col=3
        )
        
        # Third row - additional analysis
        # Cluster stability (silhouette scores)
        silhouette_scores = []
        for method in methods:
            if 'silhouette_score' in cluster_results[method]['metrics']:
                silhouette_scores.append({
                    'method': method,
                    'score': cluster_results[method]['metrics']['silhouette_score']
                })
        
        if silhouette_scores:
            fig.add_trace(
                go.Bar(
                    x=[s['method'] for s in silhouette_scores],
                    y=[s['score'] for s in silhouette_scores],
                    name='Stability'
                ),
                row=3, col=1
            )
        
        # 3D visualization of communities
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features)
        
        fig.add_trace(
            go.Scatter3d(
                x=features_3d[:, 0],
                y=features_3d[:, 1],
                z=features_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=cluster_results['louvain']['clusters']['cluster'],
                    colorscale='Viridis'
                )
            ),
            row=3, col=2
        )
        
        # Noise analysis (DBSCAN)
        noise_stats = features[dbscan_clusters['noise']].describe()
        fig.add_trace(
            go.Bar(
                x=noise_stats.index,
                y=noise_stats.values,
                name='Noise Stats'
            ),
            row=3, col=3
        )
        
        fig.update_layout(
            height=1800,
            title_text="Enhanced Clustering Analysis",
            showlegend=True
        )
        
        fig.write_html(save_path)
        return fig
    
    def calculate_period_statistics(self, regime_stats, transitions, data):
        """Calculate comprehensive statistics for a period"""
        stats = {
            'regime_distribution': {
                regime: (count / sum(regime_stats.values()) * 100)
                for regime, count in regime_stats.items()
            },
            'transitions': {
                'avg_duration': transitions['duration_days'].mean(),
                'n_transitions': len(transitions),
                'most_common': transitions.groupby(['from_regime', 'to_regime'])
                             .size().nlargest(5).to_dict()
            },
            'market_stats': {
                'n_cryptocurrencies': len(data.index.get_level_values('symbol').unique()),
                'avg_daily_volume': data['volume'].mean(),
                'volatility': data['close'].pct_change().std() * np.sqrt(252)
            }
        }
        return stats
    
    def create_period_visualizations(self, period_name, period_dir, 
                                   regime_stats, corr_matrix, transitions):
        """Create all visualizations for a period"""
        print("\nCreating visualizations...")
        
        try:
            self.plot_market_regime_distribution(
                regime_stats,
                period_dir / f"market_regime_distribution_{period_name}.html"
            )
            
            self.plot_regime_correlations(
                corr_matrix,
                period_dir / f"regime_correlations_{period_name}.html"
            )
            
            if not transitions.empty:
                self.plot_market_transitions(
                    transitions,
                    period_dir / f"market_transitions_{period_name}.html"
                )
        except Exception as e:
            print(f"Error creating visualizations for {period_name}: {e}")
    
    def run_period_analysis(self, periods: List[Dict[str, str]]):
        """Run analysis for multiple time periods"""
        try:
            print("Loading complete market data...")
            merged_df = self.merger.merge_crypto_data()
            
            all_period_stats = {}  # Store statistics for comparison
            
            for period in periods:
                period_name = period['name']
                start_date = pd.Timestamp(period['start'])
                end_date = pd.Timestamp(period['end'])
                
                print(f"\n{'='*50}")
                print(f"Analyzing period: {period_name}")
                print(f"From {start_date} to {end_date}")
                print('='*50)
                
                # Create period directory
                period_dir = self.plots_dir / period_name
                period_dir.mkdir(parents=True, exist_ok=True)
                
                # Filter data for period
                period_data = merged_df[
                    (merged_df.index.get_level_values('timestamp') >= start_date) &
                    (merged_df.index.get_level_values('timestamp') <= end_date)
                ]
                
                if period_data.empty:
                    print(f"No data available for period {period_name}")
                    continue
                
                try:
                    # Run analysis for period
                    print("\nAnalyzing market regimes...")
                    all_regimes, regime_stats = self.analyze_market_regimes(period_data)
                    
                    print("\nAnalyzing regime correlations...")
                    corr_matrix = self.analyze_regime_correlations(all_regimes)
                    
                    print("\nAnalyzing market transitions...")
                    market_transitions = self.analyze_market_transitions(all_regimes)
                    
                    if not market_transitions.empty:
                        stats = self.calculate_period_statistics(
                            regime_stats, market_transitions, period_data
                        )
                        all_period_stats[period_name] = stats
                        
                        # Create visualizations
                        self.create_period_visualizations(
                            period_name, period_dir, regime_stats, 
                            corr_matrix, market_transitions
                        )
                    else:
                        print(f"No transitions found for period {period_name}")
                    
                    print("\nPerforming cluster analysis...")
                    cluster_results, features = self.analyze_market_clusters(period_data, period_name)

                    # Create clustering visualizations
                    self.plot_louvain_network(
                        nx.Graph(corr_matrix),
                        cluster_results['louvain']['clusters']['cluster'],
                        features,
                        period_dir / f"louvain_network_{period_name}.html"
                    )
                    
                    self.plot_enhanced_clustering_comparison(
                        cluster_results,
                        features,
                        period_dir / f"enhanced_clustering_comparison_{period_name}.html"
                    )
                    
                    print("\nPerforming LSTM analysis...")
                    lstm_model, lstm_results = self.analyze_with_lstm(
                        period_data, 
                        all_regimes[list(all_regimes.keys())[0]],  # Use first symbol's regimes
                        period_name
                    )
                    
                except Exception as e:
                    print(f"Error analyzing period {period_name}: {e}")
                    continue
            
            if all_period_stats:
                # Create comparison visualizations
                print("\nCreating period comparisons...")
                self.plot_period_comparison(all_period_stats)
            
        except Exception as e:
            print(f"Error during period analysis: {e}")

    def plot_period_comparison(self, all_period_stats: Dict):
        """Create visualizations comparing different time periods"""
        try:
            comparison_dir = self.plots_dir / "comparisons"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comparison figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Regime Distribution by Period',
                    'Market Statistics Comparison',
                    'Transition Patterns',
                    'Regime Stability'
                )
            )
            
            # Plot regime distributions
            for period, stats in all_period_stats.items():
                fig.add_trace(
                    go.Bar(
                        name=period,
                        x=list(stats['regime_distribution'].keys()),
                        y=list(stats['regime_distribution'].values()),
                        text=[f"{v:.1f}%" for v in stats['regime_distribution'].values()],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
            
            # Plot market statistics
            market_stats = pd.DataFrame({
                period: {
                    'Volatility': stats['market_stats']['volatility'],
                    'Cryptocurrencies': stats['market_stats']['n_cryptocurrencies'],
                    'Avg Volume': stats['market_stats']['avg_daily_volume']
                }
                for period, stats in all_period_stats.items()
            }).T
            
            fig.add_trace(
                go.Heatmap(
                    z=market_stats.values,
                    x=market_stats.columns,
                    y=market_stats.index,
                    colorscale='Viridis'
                ),
                row=1, col=2
            )
            
            # Plot transition patterns
            transition_stats = pd.DataFrame({
                period: {
                    'Avg Duration': stats['transitions']['avg_duration'],
                    'N Transitions': stats['transitions']['n_transitions']
                }
                for period, stats in all_period_stats.items()
            }).T
            
            fig.add_trace(
                go.Bar(
                    x=transition_stats.index,
                    y=transition_stats['Avg Duration'],
                    name='Avg Duration'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=transition_stats.index,
                    y=transition_stats['N Transitions'],
                    name='N Transitions'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=1200,
                title_text="Period Comparison Analysis",
                showlegend=True,
                barmode='group'
            )
            
            fig.write_html(comparison_dir / "period_comparison.html")
            
            # Save comparison statistics
            with open(comparison_dir / "period_comparison_stats.txt", 'w') as f:
                f.write("Period Comparison Statistics\n")
                f.write("="*30 + "\n\n")
                
                for period, stats in all_period_stats.items():
                    f.write(f"\n{period}\n{'-'*len(period)}\n")
                    f.write(f"Number of cryptocurrencies: {stats['market_stats']['n_cryptocurrencies']}\n")
                    f.write(f"Average daily volume: {stats['market_stats']['avg_daily_volume']:.2f}\n")
                    f.write(f"Market volatility: {stats['market_stats']['volatility']:.2%}\n")
                    f.write(f"Number of regime transitions: {stats['transitions']['n_transitions']}\n")
                    f.write(f"Average regime duration: {stats['transitions']['avg_duration']:.1f} days\n")
                    
                    f.write("\nMost common transitions:\n")
                    for (from_regime, to_regime), count in stats['transitions']['most_common'].items():
                        f.write(f"  {from_regime} -> {to_regime}: {count}\n")
                    
                    f.write("\n")
        
        except Exception as e:
            print(f"Error creating period comparison: {e}")
            raise

    def analyze_with_lstm(self, data: pd.DataFrame, regimes: pd.DataFrame, period_name: str):
        """Analyze market regimes using LSTM"""
        print("\nTraining LSTM model for regime prediction...")
        
        # Create directory for LSTM results
        lstm_dir = self.plots_dir / period_name / "lstm"
        lstm_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            model, results = predict_regimes(data, regimes, str(lstm_dir))
            
            # Save results
            with open(lstm_dir / "model_results.txt", 'w') as f:
                f.write("LSTM Model Results\n")
                f.write("=================\n\n")
                f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"F1 Score: {results['f1_score']:.4f}\n")
            
            return model, results
        except Exception as e:
            print(f"Error in LSTM analysis: {str(e)}")
            return None, None

def main():
    analyzer = MarketAnalyzer()
    
    # Define analysis periods
    periods = [
        {
            'start': '2023-01-01',
            'end': '2023-12-31',
            'name': '2023'
        },
        {
            'start': '2024-01-01',
            'end': '2024-01-25',  # Or current date
            'name': '2024'
        },
        {
            'start': '2020-01-01',  # Full dataset
            'end': '2024-01-25',
            'name': 'full_period'
        }
    ]
    
    analyzer.run_period_analysis(periods)

if __name__ == "__main__":
    main() 