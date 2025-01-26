import argparse
from market_analysis import MarketAnalyzer
from lstm_analysis import LSTMVisualizer
from src.models.lstm_forecasting import main as run_lstm
from src.data_processing.data_fetching import CryptoDataFetcher


def run_data_fetching(start_date="2020-01-01 00:00:00", end_date="2024-12-31 00:00:00", top_n=100):
    """
    Fetch cryptocurrency data from Binance API.
    
    Parameters:
    -----------
    start_date : str
        Start date for data collection in format "YYYY-MM-DD HH:MM:SS"
    end_date : str
        End date for data collection in format "YYYY-MM-DD HH:MM:SS"
    top_n : int
        Number of top cryptocurrencies by volume to analyze
    """
    print(f"Fetching data for top {top_n} cryptocurrencies from {start_date} to {end_date}")
    fetcher = CryptoDataFetcher()
    fetcher.fetch_all_crypto_data(start_date, end_date, limit=top_n)


def run_market_analysis(period=None):
    """
    Run market regime analysis for specified period(s).
    
    Parameters:
    -----------
    period : str or dict or None
        If str: one of "2023", "2024", "full_period"
        If dict: custom period with 'start', 'end', and 'name' keys
        If None: run analysis for all predefined periods
    """
    analyzer = MarketAnalyzer()

    predefined_periods = {
        '2023': {
            'start': '2023-01-01',
            'end': '2023-12-31',
            'name': '2023'
        },
        '2024': {
            'start': '2024-01-01',
            'end': '2024-12-31',
            'name': '2024'
        },
        'full_period': {
            'start': '2020-01-01',
            'end': '2024-12-31',
            'name': 'full_period'
        }
    }

    if isinstance(period, str):
        if period in predefined_periods:
            analyzer.run_period_analysis([predefined_periods[period]])
        else:
            raise ValueError(f"Invalid period. Choose from {list(predefined_periods.keys())}")
    elif isinstance(period, dict):
        analyzer.run_period_analysis([period])
    else:
        analyzer.run_period_analysis(list(predefined_periods.values()))


def run_lstm_analysis():
    """Run LSTM analysis and generate visualizations."""
    print("Running LSTM analysis...")
    visualizer = LSTMVisualizer()
    results = run_lstm()
    
    if results:
        visualizer.create_summary_dashboard(results)
        for symbol, res in results.items():
            if 'predictions' in res and 'true_values' in res:
                visualizer.plot_prediction_performance(
                    res['true_values'],
                    res['predictions'],
                    symbol,
                    res.get('training_history')
                )
    return


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Market Regime Analysis')
    
    # Data fetching arguments
    parser.add_argument('--fetch_data', type=bool, default=False,
                      help='Whether to fetch new data from Binance API')
    parser.add_argument('--start_date', type=str, default="2020-01-01 00:00:00",
                      help='Start date for data collection (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_date', type=str, default="2024-12-31 00:00:00",
                      help='End date for data collection (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--top_n', type=int, default=100,
                      help='Number of top cryptocurrencies to analyze')
    
    # Analysis arguments
    parser.add_argument('--market_analysis', type=bool, default=False,
                      help='Whether to run market regime analysis')
    parser.add_argument('--period', type=str, default=None,
                      choices=['2023', '2024', 'full_period', None],
                      help='Period for market analysis')
    parser.add_argument('--lstm_analysis', type=bool, default=False,
                      help='Whether to run LSTM analysis')
    
    return parser.parse_args()


def main():
    """
    Main entry point for running the analysis pipeline.
    
    Command line arguments:
    ----------------------
    --fetch_data : bool
        Whether to fetch new data from Binance API
    --start_date : str
        Start date for data collection (default: "2020-01-01 00:00:00")
    --end_date : str
        End date for data collection (default: "2024-12-31 00:00:00")
    --top_n : int
        Number of top cryptocurrencies to analyze (default: 100)
    --market_analysis : bool
        Whether to run market regime analysis
    --period : str
        Period for market analysis (2023, 2024, full_period, or None)
    --lstm_analysis : bool
        Whether to run LSTM analysis
    """
    args = parse_args()
    
    if args.fetch_data:
        run_data_fetching(
            start_date=args.start_date,
            end_date=args.end_date,
            top_n=args.top_n
        )

    if args.market_analysis:
        run_market_analysis(period=args.period)

    if args.lstm_analysis:
        run_lstm_analysis()


if __name__ == "__main__":
    main()
