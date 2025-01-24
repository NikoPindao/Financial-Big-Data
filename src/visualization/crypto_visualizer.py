import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import ta

class CryptoVisualizer:
    def __init__(self):
        """Initialize the CryptoVisualizer with data directory paths."""
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.plots_dir = self.data_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_symbol_data(self, symbol):
        """Load data for a specific symbol from parquet file."""
        file_path = self.raw_dir / f"{symbol.replace('/', '_')}_data.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for {symbol}")
        return pd.read_parquet(file_path)

    def plot_price_volume(self, symbol, days=30):
        """
        Create an interactive price and volume chart for the specified symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            days (int): Number of recent days to plot
        """
        df = self.load_symbol_data(symbol)
        
        # Filter for the last n days
        df = df.sort_values('timestamp')
        last_date = df['timestamp'].max()
        start_date = last_date - timedelta(days=days)
        df = df[df['timestamp'] >= start_date]

        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=(f'{symbol} Price', 'Volume'),
                           row_heights=[0.7, 0.3])

        # Add candlestick
        fig.add_trace(go.Candlestick(x=df['timestamp'],
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='OHLC'),
                     row=1, col=1)

        # Add volume bar chart
        fig.add_trace(go.Bar(x=df['timestamp'],
                            y=df['volume'],
                            name='Volume'),
                     row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol} Price and Volume Chart (Last {days} Days)',
            xaxis_rangeslider_visible=False,
            height=800
        )

        # Save the plot
        fig.write_html(self.plots_dir / f"{symbol.replace('/', '_')}_chart.html")
        return fig

    def plot_market_overview(self, top_n=10):
        """
        Create a market overview showing performance of top n cryptocurrencies.
        
        Args:
            top_n (int): Number of top cryptocurrencies to include
        """
        # Load symbols from json
        with open(self.raw_dir / "top_symbols.json", 'r') as f:
            symbols = json.load(f)[:top_n]

        # Calculate daily returns for each symbol
        returns_data = []
        for symbol in symbols:
            df = self.load_symbol_data(symbol)
            df = df.sort_values('timestamp')
            daily_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
            returns_data.append({
                'symbol': symbol.replace('/USDT', ''),
                'return': daily_return
            })

        # Create returns comparison chart
        df_returns = pd.DataFrame(returns_data)
        fig = go.Figure(data=[
            go.Bar(x=df_returns['symbol'],
                  y=df_returns['return'],
                  text=df_returns['return'].round(2),
                  textposition='auto')
        ])

        fig.update_layout(
            title='Performance Comparison (%)',
            xaxis_title='Cryptocurrency',
            yaxis_title='Return (%)',
            height=600
        )

        # Save the plot
        fig.write_html(self.plots_dir / "market_overview.html")
        return fig

    def plot_correlation_matrix(self, top_n=10):
        """
        Create a correlation matrix heatmap for top n cryptocurrencies.
        
        Args:
            top_n (int): Number of top cryptocurrencies to include
        """
        # Load symbols
        with open(self.raw_dir / "top_symbols.json", 'r') as f:
            symbols = json.load(f)[:top_n]

        # Calculate returns for correlation
        returns_dict = {}
        for symbol in symbols:
            df = self.load_symbol_data(symbol)
            df = df.sort_values('timestamp')
            returns = df['close'].pct_change()
            returns_dict[symbol.replace('/USDT', '')] = returns

        # Create correlation matrix
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
            title='Correlation Matrix',
            height=800,
            width=800
        )

        # Save the plot
        fig.write_html(self.plots_dir / "correlation_matrix.html")
        return fig

    def analyze_single_crypto(self, symbol, days=30, include_indicators=True):
        """
        Create a detailed analysis visualization for a single cryptocurrency.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            days (int): Number of recent days to analyze
            include_indicators (bool): Whether to include technical indicators
        
        Returns:
            go.Figure: Interactive plotly figure
        """
        # Load and prepare data
        df = self.load_symbol_data(symbol)
        df = df.sort_values('timestamp')
        
        # Filter for the specified time period
        last_date = df['timestamp'].max()
        start_date = last_date - timedelta(days=days)
        df = df[df['timestamp'] >= start_date].copy()
        
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=24).std() * np.sqrt(24 * 365)  # Annualized
        
        # Calculate technical indicators
        if include_indicators:
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_high'] = bollinger.bollinger_hband()
            df['BB_low'] = bollinger.bollinger_lband()
            df['BB_mid'] = bollinger.bollinger_mavg()
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=(
                f'{symbol} Price and Bollinger Bands',
                'Volume',
                'RSI',
                'MACD'
            )
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        if include_indicators:
            # Add Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['BB_high'],
                    name='BB Upper',
                    line=dict(color='gray', dash='dash'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['BB_low'],
                    name='BB Lower',
                    line=dict(color='gray', dash='dash'),
                    opacity=0.5,
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # Add volume
        colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        if include_indicators:
            # Add RSI
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['RSI'],
                    name='RSI'
                ),
                row=3, col=1
            )
            # Add RSI lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # Add MACD
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['MACD'],
                    name='MACD'
                ),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['MACD_signal'],
                    name='Signal'
                ),
                row=4, col=1
            )
        
        # Add statistics annotation
        stats_text = f"""
        Current Price: ${df['close'].iloc[-1]:.2f}
        24h Change: {((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100:.1f}%
        30d Volatility: {df['volatility'].iloc[-1] * 100:.1f}%
        Volume: {df['volume'].iloc[-1]:.0f}
        """
        
        fig.add_annotation(
            text=stats_text,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=1.02,
            y=0.98,
            bordercolor='black',
            borderwidth=1,
            bgcolor='white'
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Analysis (Last {days} Days)',
            xaxis_rangeslider_visible=False,
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save the plot
        fig.write_html(self.plots_dir / f"{symbol.replace('/', '_')}_analysis.html")
        return fig

def main():
    """Example usage of the CryptoVisualizer."""
    visualizer = CryptoVisualizer()
    visualizer.analyze_single_crypto("BTC/USDT", days=1825)  # 5 years of data

if __name__ == "__main__":
    main() 