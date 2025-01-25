import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class RegimeVisualizer:
    def __init__(self):
        self.regime_colors = {
            'stable_bull': 'rgba(0, 255, 0, 0.2)',
            'volatile_bull': 'rgba(0, 255, 0, 0.4)',
            'stable_bear': 'rgba(255, 0, 0, 0.2)',
            'volatile_bear': 'rgba(255, 0, 0, 0.4)',
            'consolidation': 'rgba(128, 128, 128, 0.2)'
        }
    
    def plot_regime_analysis(self, df: pd.DataFrame, regimes: pd.DataFrame, 
                           symbol: str, save_path: str = None):
        """Create comprehensive regime analysis visualization"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price and Regimes', 'Volume', 'Regime Changes'),
            row_heights=[0.5, 0.25, 0.25]
        )

        # Price candlesticks
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
        for regime in self.regime_colors:
            mask = regimes['regime'] == regime
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=regimes[mask].index,
                        y=df['high'],
                        fill='tonexty',
                        fillcolor=self.regime_colors[regime],
                        name=regime,
                        showlegend=True
                    ),
                    row=1, col=1
                )

        # Volume
        colors = ['red' if o > c else 'green' 
                 for o, c in zip(df['open'], df['close'])]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )

        # Regime changes
        regime_changes = regimes['regime'].ne(regimes['regime'].shift()).cumsum()
        fig.add_trace(
            go.Scatter(
                x=regimes.index,
                y=regime_changes,
                name='Regime Changes'
            ),
            row=3, col=1
        )

        fig.update_layout(
            title=f'{symbol} Market Regime Analysis',
            height=1200,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
        
        return fig

    def plot_cluster_comparison(self, dfs: Dict[str, pd.DataFrame], 
                              cluster_labels: Dict[str, str], 
                              save_path: str = None):
        """Compare cryptocurrencies in the same cluster"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Normalized Prices', 'Trading Volume'),
            row_heights=[0.7, 0.3]
        )

        # Normalize prices for comparison
        for symbol, df in dfs.items():
            norm_price = df['close'] / df['close'].iloc[0] * 100
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=norm_price,
                    name=f"{symbol} ({cluster_labels[symbol]})",
                    mode='lines'
                ),
                row=1, col=1
            )

            # Add volume as area plot
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['volume'],
                    name=f"{symbol} Volume",
                    fill='tonexty',
                    opacity=0.3
                ),
                row=2, col=1
            )

        fig.update_layout(
            title='Cluster Comparison - Normalized Prices and Volume',
            height=1000,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
        
        return fig 

    def plot_regime_transitions(self, transitions_df: pd.DataFrame, save_path: str = None):
        """Visualize regime transitions"""
        if transitions_df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Regime Duration Distribution',
                'Transition Probabilities',
                'Regime Frequency',
                'Transition Types'
            )
        )
        
        # Duration distribution
        fig.add_trace(
            go.Box(
                y=transitions_df['duration_days'],
                name='Regime Duration (days)',
                boxpoints='all'
            ),
            row=1, col=1
        )
        
        # Transition probabilities heatmap
        probs = transitions_df.attrs.get('transition_probabilities', {})
        if probs:
            regimes = sorted(transitions_df['from_regime'].unique())
            prob_matrix = np.zeros((len(regimes), len(regimes)))
            for i, from_regime in enumerate(regimes):
                for j, to_regime in enumerate(regimes):
                    key = f"{from_regime}_to_{to_regime}"
                    prob_matrix[i, j] = probs.get(key, 0)
            
            fig.add_trace(
                go.Heatmap(
                    z=prob_matrix,
                    x=regimes,
                    y=regimes,
                    colorscale='Viridis'
                ),
                row=1, col=2
            )
        
        # Regime frequency
        regime_counts = transitions_df['from_regime'].value_counts()
        fig.add_trace(
            go.Bar(
                x=regime_counts.index,
                y=regime_counts.values,
                name='Regime Frequency'
            ),
            row=2, col=1
        )
        
        # Transition types
        transition_types = pd.DataFrame({
            'Type': ['Volatility Increase', 'Trend Change'],
            'Count': [
                transitions_df['is_volatility_increase'].sum(),
                transitions_df['is_trend_change'].sum()
            ]
        })
        fig.add_trace(
            go.Bar(
                x=transition_types['Type'],
                y=transition_types['Count'],
                name='Transition Types'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Regime Transition Analysis",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig 