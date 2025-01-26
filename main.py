from market_analysis import MarketAnalyzer
from lstm_analysis import LSTMVisualizer
from src.models.lstm_forecasting import main as run_lstm
from src.data_processing.data_fetching import main as run_data_fetching

def run_market_analysis():
    analyzer = MarketAnalyzer()
    
    periods = [
        {   #For 2023
            'start': '2023-01-01',
            'end': '2023-12-31',
            'name': '2023'
        },
        {   #For 2024
            'start': '2024-01-01',
            'end': '2024-12-31',  
            'name': '2024'
        },
        {  #Full period
            'start': '2020-01-01', 
            'end': '2024-12-31',
            'name': 'full_period'
        }
    ]
    analyzer.run_period_analysis(periods)
    return

def run_lstm_analysis():
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

def main():
    #Run data fetching first from the last 4 years (2020-2024) and save it to data/raw as parquet files
    run_data_fetching()
    #Run market analysis to get the market regimes and output it to data/plots/market_analysis
    run_market_analysis()
    #Run lstm analysis to get the lstm predictions and output it to data/plots/lstm_analysis
    run_lstm_analysis()

if __name__ == "__main__":
    main()
