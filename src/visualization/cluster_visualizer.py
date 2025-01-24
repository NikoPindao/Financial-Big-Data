import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

class ClusterVisualizer:
    def __init__(self):
        self.plots_dir = Path("data/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_cluster_analysis(self, df, clusters_df, symbol, method='louvain'):
        """
        Create visualization for clustering analysis.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} Price and Clusters ({method})',
                'Cluster Metrics'
            ),
            row_heights=[0.7, 0.3]
        )

        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add cluster backgrounds
        unique_clusters = clusters_df['cluster'].unique()
        colors = [f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.1)' 
                 for _ in range(len(unique_clusters))]
        
        for cluster, color in zip(unique_clusters, colors):
            mask = clusters_df['cluster'] == cluster
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=clusters_df[mask].index,
                        y=df['high'],
                        fill='tonexty',
                        fillcolor=color,
                        name=f'Cluster {cluster}',
                        showlegend=True
                    ),
                    row=1, col=1
                )

        # Metrics plot
        if 'silhouette' in clusters_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=clusters_df.index,
                    y=clusters_df['silhouette'],
                    name='Silhouette Score'
                ),
                row=2, col=1
            )

        fig.update_layout(
            title=f'{symbol} Cluster Analysis using {method}',
            height=1000,
            showlegend=True
        )

        # Save the plot
        fig.write_html(self.plots_dir / f"{symbol.replace('/', '_')}_{method}_clusters.html")
        return fig

    def plot_optimal_clusters(self, n_clusters, silhouette_scores):
        """
        Plot silhouette scores for different numbers of clusters.
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=list(range(2, len(silhouette_scores) + 2)),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score'
            )
        )
        
        fig.update_layout(
            title='Optimal Number of Clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='Silhouette Score',
            height=600
        )
        
        # Save the plot
        fig.write_html(self.plots_dir / "optimal_clusters.html")
        return fig

    def plot_cluster_comparison(self, df, clusters_dict, symbol):
        """
        Compare different clustering methods.
        
        Args:
            df: Original price data
            clusters_dict: Dictionary of clustering results {method: clusters_df}
        """
        n_methods = len(clusters_dict)
        fig = make_subplots(
            rows=n_methods, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f'{method} Clustering' for method in clusters_dict.keys()]
        )
        
        for i, (method, clusters_df) in enumerate(clusters_dict.items(), 1):
            # Add price data
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=f'Price ({method})'
                ),
                row=i, col=1
            )
            
            # Add cluster backgrounds
            unique_clusters = clusters_df['cluster'].unique()
            colors = [f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.1)' 
                     for _ in range(len(unique_clusters))]
            
            for cluster, color in zip(unique_clusters, colors):
                mask = clusters_df['cluster'] == cluster
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=clusters_df[mask].index,
                            y=df['high'],
                            fill='tonexty',
                            fillcolor=color,
                            name=f'{method} Cluster {cluster}',
                            showlegend=True
                        ),
                        row=i, col=1
                    )
        
        fig.update_layout(
            title=f'{symbol} Clustering Methods Comparison',
            height=300 * n_methods,
            showlegend=True
        )
        
        # Save the plot
        fig.write_html(self.plots_dir / f"{symbol.replace('/', '_')}_cluster_comparison.html")
        return fig 