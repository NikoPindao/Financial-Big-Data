import pandas as pd
import ccxt
from datetime import datetime
import time
import json
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_data_fetching.log'),
        logging.StreamHandler()
    ]
)

class CryptoDataFetcher:
    def __init__(self, exchange_id='binance', rate_limit=1200):
        self.exchange = ccxt.binance({
            "rateLimit": rate_limit,
            "enableRateLimit": True
        })
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def get_top_symbols(self, limit=100):
        """
        Calcule les top 100 symboles par volume
        """
        logging.info(f"Fetching top {limit} symbols by volume...")
        
        try:
            markets = self.exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
            
            volumes = {}
            for pair in tqdm(usdt_pairs, desc="Fetching volumes"):
                try:
                    ticker = self.exchange.fetch_ticker(pair)
                    volumes[pair] = ticker['quoteVolume']
                    time.sleep(self.exchange.rateLimit / 1000)
                except Exception as e:
                    logging.warning(f"Error fetching volume for {pair}: {e}")
                    volumes[pair] = 0
            
            top_pairs = sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:limit]
            symbols = [pair[0] for pair in top_pairs]
            with open(self.raw_dir / "top_symbols.json", "w") as f:
                json.dump(symbols, f)
            
            return symbols
            
        except Exception as e:
            logging.error(f"Error getting top symbols: {e}")
            raise

    def load_symbol_data(self, symbol):
        file_path = self.raw_dir / f"{symbol.replace('/', '_')}_data.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for {symbol} at {file_path}")
        
        df = pd.read_parquet(file_path)
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_ohlcv_data(self, symbol, timeframe, start_date, end_date, retries=3):
        """
        Fetch OHLCV data for a single symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe for candles (e.g., '1h')
            start_date (str): Start date in 'YYYY-MM-DD HH:MM:SS' format
            end_date (str): End date in 'YYYY-MM-DD HH:MM:SS' format
            retries (int): Number of retry attempts
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        since = int(datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        end_time = int(datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        all_data = []
        while since < end_time:
            for attempt in range(retries):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe=timeframe, 
                        since=since, 
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data += ohlcv
                    since = ohlcv[-1][0] + 1
                    time.sleep(self.exchange.rateLimit / 1000)
                    break
                    
                except Exception as e:
                    if attempt == retries - 1:
                        logging.error(f"Failed to fetch {symbol} after {retries} attempts: {e}")
                        return pd.DataFrame()
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            all_data, 
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_all_crypto_data(self, start_date, end_date, timeframe="1h", limit=100):
        """
        Fetch data for all top cryptocurrencies.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD HH:MM:SS' format
            end_date (str): End date in 'YYYY-MM-DD HH:MM:SS' format
            timeframe (str): Timeframe for candles (default: '1h')
            limit (int): Number of top cryptocurrencies to fetch (default: 100)
        """
        symbols = self.get_top_symbols(limit)
        logging.info(f"Starting data collection for {len(symbols)} symbols")
        
        for symbol in tqdm(symbols, desc="Fetching cryptocurrency data"):
            try:
                df = self.fetch_ohlcv_data(symbol, timeframe, start_date, end_date)
                
                if not df.empty:
                    # Save to parquet file
                    filename = self.raw_dir / f"{symbol.replace('/', '_')}_data.parquet"
                    df.to_parquet(filename)
                    logging.info(f"Successfully saved data for {symbol}")
                else:
                    logging.warning(f"No data retrieved for {symbol}")
                    
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")

def engineer_features(df):
    """
    Crée les features pour le market regime, à modifier si jamais
    """
    features = pd.DataFrame()
    
    # Returns
    features['returns'] = df['close'].pct_change()
    
    # Volatility (using different windows)
    features['volatility_1d'] = features['returns'].rolling(24).std()  # 24 hours
    features['volatility_1w'] = features['returns'].rolling(168).std() # 1 week
    
    # Volume features
    features['volume_change'] = df['volume'].pct_change()
    features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(168).mean()
    
    # Price momentum
    features['momentum_1d'] = df['close'].pct_change(24)
    features['momentum_1w'] = df['close'].pct_change(168)
    
    return features
