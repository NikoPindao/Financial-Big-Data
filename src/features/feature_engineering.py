from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

class FeatureEngineer:
    """Class for cryptocurrency feature engineering"""
    
    def __init__(self):
        self.default_windows = [24, 168]  # 1 day, 1 week
        self.volume_ma_window = 24  # 1 day
    
    def calculate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price-based features"""
        returns = df['close'].pct_change()
        
        features = {
            'volatility': returns.std() * np.sqrt(252),
            'daily_return': returns.mean() * 252,
            'skewness': skew(returns.dropna()),
            'kurtosis': kurtosis(returns.dropna()),
            'high_low_ratio': (df['high'] / df['low']).mean(),
            'price_range': ((df['high'] - df['low']) / df['close']).mean()
        }
        
        # Add momentum features for different windows
        for window in self.default_windows:
            features[f'momentum_{window}h'] = df['close'].pct_change(window).mean()
        
        return features
    
    def calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based features"""
        volume_ma = df['volume'].rolling(window=self.volume_ma_window).mean()
        volume_ratio = df['volume'] / volume_ma
        
        return {
            'volume_trend': volume_ratio.mean(),
            'volume_volatility': volume_ratio.std(),
            'volume_price_corr': df['volume'].corr(df['close']),
            'volume_momentum': df['volume'].pct_change().mean() * 252
        }
    
    def calculate_regime_features(self, regimes: pd.DataFrame, transitions: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime-based features"""
        features = {
            'regime_changes': len(regimes['regime'].unique()),
            'avg_regime_duration': transitions['duration_days'].mean() if not transitions.empty else 0,
            'regime_stability': 1 - (len(transitions) / len(regimes)) if len(regimes) > 0 else 0
        }
        
        # Add regime type proportions
        total_periods = len(regimes)
        for regime_type in regimes['regime'].unique():
            features[f'regime_{regime_type}_prop'] = (regimes['regime'] == regime_type).sum() / total_periods
            
        return features
    
    def engineer_features(self, df: pd.DataFrame, regimes: pd.DataFrame, 
                         transitions: pd.DataFrame) -> pd.Series:
        """Combine all features into a single feature vector"""
        features = {}
        
        # Add all feature groups
        features.update(self.calculate_price_features(df))
        features.update(self.calculate_volume_features(df))
        features.update(self.calculate_regime_features(regimes, transitions))
        
        return pd.Series(features) 