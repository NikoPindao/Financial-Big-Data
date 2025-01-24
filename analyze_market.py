from src.visualization.crypto_visualizer import CryptoVisualizer
from src.data_processing.data_fetching import CryptoDataFetcher
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

class MarketAnalyzer:
    def __init__(self):
        self.visualizer = CryptoVisualizer()
        self.fetcher = CryptoDataFetcher()
        self.plots_dir = Path("data/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_crypto_data(self):
        """Load data for all cryptocurrencies in the test directory."""
        crypto_files = list(Path("data/raw").glob("*_data.parquet"))
        all_data = {}
        
        for file in crypto_files:
            symbol = file.stem.replace("_data", "").replace("_", "/")
            df = pd.read_parquet(file)
            df.set_index('timestamp', inplace=True)
            all_data[symbol] = df
            
        return all_data

    def plot_market_overview(self, all_data):
        """Create a market overview showing returns and volatility."""
        market_stats = []
        
        for symbol, df in all_data.items():
            # Calculate statistics
            returns = df['close'].pct_change()
            stats = {
                'symbol': symbol.replace('/USDT', ''),
                'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
                'volatility': returns.std() * np.sqrt(365 * 24) * 100,  # Annualized
                'volume_avg': df['volume'].mean(),
                'price_current': df['close'].iloc[-1],
                'market_cap': df['close'].iloc[-1] * df['volume'].mean()  # Approximate
            }
            market_stats.append(stats)
        
        stats_df = pd.DataFrame(market_stats)
        
        # 1. Returns vs Volatility Scatter Plot
        fig = px.scatter(stats_df, 
                        x='volatility', 
                        y='total_return',
                        text='symbol',
                        size='market_cap',
                        title='Cryptocurrency Risk-Return Profile',
                        labels={'volatility': 'Annualized Volatility (%)',
                               'total_return': 'Total Return (%)',
                               'market_cap': 'Approximate Market Cap'})
        
        fig.write_html(self.plots_dir / "market_risk_return.html")

    def plot_correlation_heatmap(self, all_data):
        """Create a correlation heatmap of crypto returns."""
        # Calculate daily returns for all cryptocurrencies
        returns_dict = {}
        for symbol, df in all_data.items():
            returns_dict[symbol.replace('/USDT', '')] = df['close'].pct_change()
        
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdBu'
        ))
        
        fig.update_layout(
            title='Cryptocurrency Returns Correlation Matrix',
            height=1000,
            width=1000
        )
        
        fig.write_html(self.plots_dir / "correlation_heatmap.html")

    def plot_volume_analysis(self, all_data):
        """Create volume analysis visualization."""
        volume_stats = []
        
        for symbol, df in all_data.items():
            # Calculate volume statistics
            volume_stats.append({
                'symbol': symbol.replace('/USDT', ''),
                'avg_volume': df['volume'].mean(),
                'volume_volatility': df['volume'].std() / df['volume'].mean(),
                'volume_trend': df['volume'].iloc[-30:].mean() / df['volume'].iloc[:30].mean()
            })
        
        stats_df = pd.DataFrame(volume_stats)
        
        # Volume Analysis Plot
        fig = px.scatter(stats_df,
                        x='volume_volatility',
                        y='volume_trend',
                        text='symbol',
                        size='avg_volume',
                        title='Cryptocurrency Volume Analysis',
                        labels={'volume_volatility': 'Volume Volatility (CV)',
                               'volume_trend': 'Recent Volume Trend (30-day)',
                               'avg_volume': 'Average Trading Volume'})
        
        fig.write_html(self.plots_dir / "volume_analysis.html")

    def plot_performance_comparison(self, all_data):
        """Create performance comparison chart."""
        # Calculate normalized prices for comparison
        normalized_prices = pd.DataFrame()
        
        for symbol, df in all_data.items():
            normalized_prices[symbol.replace('/USDT', '')] = df['close'] / df['close'].iloc[0] * 100
        
        # Create line plot
        fig = go.Figure()
        
        for column in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[column],
                name=column,
                mode='lines'
            ))
        
        fig.update_layout(
            title='Cryptocurrency Performance Comparison (Normalized)',
            xaxis_title='Date',
            yaxis_title='Normalized Price (%)',
            height=800
        )
        
        fig.write_html(self.plots_dir / "performance_comparison.html")

def main():
    analyzer = MarketAnalyzer()
    print("Loading cryptocurrency data...")
    all_data = analyzer.load_all_crypto_data()
    
    print("Creating market overview...")
    analyzer.plot_market_overview(all_data)
    
    print("Creating correlation heatmap...")
    analyzer.plot_correlation_heatmap(all_data)
    
    print("Creating volume analysis...")
    analyzer.plot_volume_analysis(all_data)
    
    print("Creating performance comparison...")
    analyzer.plot_performance_comparison(all_data)
    
    print("\nAnalysis complete! The following visualizations have been created:")
    print("1. market_risk_return.html - Risk-Return Profile")
    print("2. correlation_heatmap.html - Correlation Matrix")
    print("3. volume_analysis.html - Volume Analysis")
    print("4. performance_comparison.html - Performance Comparison")

if __name__ == "__main__":
    main() 