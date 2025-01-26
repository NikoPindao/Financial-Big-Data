import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from src.utils import identify_market_regime

class CryptoDataMerger:
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"

        #Create the processed directory if it doesn't exist, you can later save data there if needed
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def merge_crypto_data(self, save=False):
        crypto_files = list(self.raw_dir.glob("*_data.parquet"))
        if not crypto_files:
            raise FileNotFoundError(f"No parquet files found in {self.raw_dir}")
        
        dfs = []
        for file in tqdm(crypto_files, desc="Processing files"):
            try:
                symbol = file.stem.replace("_data", "").replace("_", "/")
                df = pd.read_parquet(file)

                #On fait le market regime par symbol (=crypto)
                df['symbol'] = symbol
                df['market_regime'] = identify_market_regime(df)
                dfs.append(df)
                
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")
                continue
                
        merged_df = pd.concat(dfs, axis=0)
        merged_df.set_index(['symbol', 'timestamp'], inplace=True)
        merged_df.sort_index(inplace=True)
        
        if save:
            output_file = self.processed_dir / "merged_crypto_data.parquet"
            merged_df.to_parquet(output_file)
            print(f"Saved merged data to {output_file}")
        
        return merged_df
    
    def load_merged_data(self):
        file_path = self.processed_dir / "merged_crypto_data.parquet"
        if not file_path.exists():
            print("Merged data file not found. Creating it now...")
            return self.merge_crypto_data()
        
        return pd.read_parquet(file_path)
    
    def get_symbol_data(self, merged_df, symbol):
        return merged_df.loc[symbol].copy()
    
    def get_date_range_data(self, merged_df, start_date=None, end_date=None):
        if start_date:
            merged_df = merged_df[merged_df.index.get_level_values('timestamp') >= start_date]
        if end_date:
            merged_df = merged_df[merged_df.index.get_level_values('timestamp') <= end_date]
        return merged_df