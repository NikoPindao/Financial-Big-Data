import networkx as nx
from community import community_louvain
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

class MarketRegimeDetector:
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.plots_dir = self.data_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_features(self, df):
        """
        Prepare and normalize features for regime detection.
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close']).diff()
        
        # Volatility features
        for window in [24, 168]:  # 1 day, 1 week
            features[f'volatility_{window}h'] = features['returns'].rolling(window).std()
        
        # Volume features
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(168).mean()
        
        # Momentum features
        for window in [24, 168]:  # 1 day, 1 week
            features[f'momentum_{window}h'] = df['close'].pct_change(window)
        
        # Normalize features
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features.fillna(0))
        return pd.DataFrame(normalized, index=features.index, columns=features.columns)

    def classify_regime(self, window_data):
        """
        Classify market regime based on returns and volatility.
        """
        returns = window_data['returns'].mean()
        volatility = window_data['volatility_24h'].mean()
        
        if returns > 0.001:  # Positive returns threshold
            if volatility > 0.02:
                return 'Volatile Bull'
            return 'Stable Bull'
        elif returns < -0.001:  # Negative returns threshold
            if volatility > 0.02:
                return 'Volatile Bear'
            return 'Stable Bear'
        else:
            if volatility > 0.02:
                return 'Volatile Consolidation'
            return 'Stable Consolidation'

    def detect_regimes(self, df, window_size=168):  # 1 week default
        """
        Enhanced regime detection with multiple features.
        """
        features = self.prepare_features(df)
        regimes = []
        
        for i in range(len(features) - window_size):
            window = features.iloc[i:i+window_size]
            
            # Create similarity network
            G = self.create_similarity_network(window)
            
            # Apply Louvain method
            communities = community_louvain.best_partition(G)
            
            # Classify regime
            regime_type = self.classify_regime(window)
            
            # Calculate regime stability
            stability = self.calculate_regime_stability(G)
            
            regimes.append({
                'timestamp': features.index[i+window_size],
                'n_communities': len(set(communities.values())),
                'regime_type': regime_type,
                'stability': stability,
                'modularity': community_louvain.modularity(communities, G),
                'avg_degree': np.mean([d for n, d in G.degree()])
            })
        
        return pd.DataFrame(regimes)

    def calculate_regime_stability(self, G):
        """
        Calculate regime stability based on network properties.
        """
        if len(G.nodes()) == 0:
            return 0
            
        clustering = nx.average_clustering(G)
        density = nx.density(G)
        return (clustering + density) / 2

    def plot_regime_analysis(self, df, regimes_df, symbol):
        """
        Create comprehensive regime analysis visualization.
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} Price and Regimes',
                'Regime Stability',
                'Number of Communities'
            ),
            row_heights=[0.5, 0.25, 0.25]
        )

        # Price chart with regime backgrounds
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

        # Add regime backgrounds
        regime_colors = {
            'Stable Bull': 'rgba(0, 255, 0, 0.1)',
            'Volatile Bull': 'rgba(0, 255, 0, 0.2)',
            'Stable Bear': 'rgba(255, 0, 0, 0.1)',
            'Volatile Bear': 'rgba(255, 0, 0, 0.2)',
            'Stable Consolidation': 'rgba(128, 128, 128, 0.1)',
            'Volatile Consolidation': 'rgba(128, 128, 128, 0.2)'
        }

        for regime_type in regime_colors:
            mask = regimes_df['regime_type'] == regime_type
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=regimes_df[mask]['timestamp'],
                        y=df['high'],
                        fill='tonexty',
                        fillcolor=regime_colors[regime_type],
                        name=regime_type,
                        showlegend=True
                    ),
                    row=1, col=1
                )

        # Stability plot
        fig.add_trace(
            go.Scatter(
                x=regimes_df['timestamp'],
                y=regimes_df['stability'],
                name='Regime Stability'
            ),
            row=2, col=1
        )

        # Communities plot
        fig.add_trace(
            go.Scatter(
                x=regimes_df['timestamp'],
                y=regimes_df['n_communities'],
                name='Number of Communities'
            ),
            row=3, col=1
        )

        fig.update_layout(
            title=f'{symbol} Market Regime Analysis',
            height=1200,
            showlegend=True
        )

        # Save the plot
        fig.write_html(self.plots_dir / f"{symbol.replace('/', '_')}_regime_analysis.html")
        return fig

def main():
    """Example usage of regime detection."""
    from data_processing.data_fetching import CryptoDataFetcher
    
    # Initialize classes
    detector = MarketRegimeDetector()
    fetcher = CryptoDataFetcher()
    
    # Load data for BTC
    df = fetcher.load_symbol_data("BTC/USDT")
    
    # Detect regimes
    regimes_df = detector.detect_regimes(df)
    
    # Plot analysis
    detector.plot_regime_analysis(df, regimes_df, "BTC/USDT")

if __name__ == "__main__":
    main() 