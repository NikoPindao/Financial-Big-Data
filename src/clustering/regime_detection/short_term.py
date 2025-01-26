from .base_detector import BaseRegimeDetector, BaseThresholds
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ShortTermThresholds(BaseThresholds):
    """Thresholds for short-term regime detection (1-7 days)"""
    
    #We use the same thresholds for all assets, change the parameters to adapt if needed
    def __init__(self):
        super().__init__(
            volatility_threshold=0.02,  # 2% daily volatility
            return_threshold=0.01,      # 1% daily return
            volume_threshold=1.5,       # 50% above average
            trend_threshold=0.6         # 60% directional consistency
        )

class ShortTermDetector(BaseRegimeDetector):
    """
    Short-term regime detector (1-7 days)
    """

    def __init__(self):
        super().__init__(
            window_size=24,  # 24 hours
            overlap=12,      # 12 hours overlap
            thresholds=ShortTermThresholds()
        )
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for short-term analysis"""
        features = pd.DataFrame(index=df.index)
        
        # Basic returns and volatility, we do a mix between 24H and 1H
        returns = df['close'].pct_change()
        features['hourly_returns'] = returns
        features['daily_returns'] = df['close'].pct_change(24)
        features['volatility'] = returns.rolling(self.window_size).std() * np.sqrt(24)
        features['volume_ma'] = df['volume'].rolling(self.window_size).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        features['trend_direction'] = np.sign(features['daily_returns'])
        features['trend_strength'] = features['trend_direction'].rolling(self.window_size).mean().abs()
        features['momentum'] = df['close'].pct_change(self.window_size)
        
        return features
    
    def detect_regime(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect short-term market regime"""
        features = self.prepare_features(df)
        
        regimes = pd.DataFrame(index=features.index)
        
        bull_conditions = (
            (features['daily_returns'] > self.thresholds.return_threshold) &
            (features['trend_strength'] > self.thresholds.trend_threshold)
        )
        bear_conditions = (
            (features['daily_returns'] < -self.thresholds.return_threshold) &
            (features['trend_strength'] > self.thresholds.trend_threshold)
        )
        
        high_vol = features['volatility'] > self.thresholds.volatility_threshold
        
        regimes['regime'] = 'consolidation'  
        regimes.loc[bull_conditions & ~high_vol, 'regime'] = 'stable_bull'
        regimes.loc[bull_conditions & high_vol, 'regime'] = 'volatile_bull'
        regimes.loc[bear_conditions & ~high_vol, 'regime'] = 'stable_bear'
        regimes.loc[bear_conditions & high_vol, 'regime'] = 'volatile_bear'
        
        # Calculate confidence metrics
        confidence = {
            'trend_confidence': features['trend_strength'].mean(),
            'volatility_stability': 1 - (features['volatility'].std() / features['volatility'].mean()),
            'volume_support': (features['volume_ratio'] > 1).mean()
        }
        
        return regimes, confidence

    def analyze_regime_transitions(self, regimes: pd.DataFrame) -> pd.DataFrame:
        """Analyze regime transitions and their characteristics"""
        transitions_list = []
        current_regime = None
        start_time = None
        
        for time, regime in regimes['regime'].items():
            if regime != current_regime:
                if current_regime is not None:
                    duration = (time - start_time).total_seconds() / 3600  # hours
                    transitions_list.append({
                        'from_regime': current_regime,
                        'to_regime': regime,
                        'start_time': start_time,
                        'end_time': time,
                        'duration': duration,
                        'duration_days': duration / 24
                    })
                current_regime = regime
                start_time = time
        
        transitions_df = pd.DataFrame(transitions_list)
        
        if not transitions_df.empty:
            # Add transition statistics
            transitions_df['is_volatility_increase'] = transitions_df['to_regime'].str.contains('volatile')
            transitions_df['is_trend_change'] = (
                (transitions_df['from_regime'].str.contains('bull') & transitions_df['to_regime'].str.contains('bear')) |
                (transitions_df['from_regime'].str.contains('bear') & transitions_df['to_regime'].str.contains('bull'))
            )
            
            #We calculate the transition probabilities
            transition_probs = {}
            for from_regime in transitions_df['from_regime'].unique():
                from_mask = transitions_df['from_regime'] == from_regime
                for to_regime in transitions_df['to_regime'].unique():
                    to_mask = transitions_df['to_regime'] == to_regime
                    prob = (from_mask & to_mask).sum() / from_mask.sum()
                    transition_probs[f"{from_regime}_to_{to_regime}"] = prob
            
            #Final summary
            transitions_df.attrs['transition_probabilities'] = transition_probs
            transitions_df.attrs['avg_regime_duration'] = transitions_df['duration'].mean()
            transitions_df.attrs['most_common_transition'] = (
                transitions_df.groupby(['from_regime', 'to_regime'])
                .size()
                .sort_values(ascending=False)
                .index[0]
            )
        
        return transitions_df 