import numpy as np
import pandas as pd

def identify_market_regime(df, volatility_window=24, momentum_window=24, 
                            volatility_percentile=80, momentum_percentile=90):
    """
    Identify market regime dynamically:
    
    Args:
        df: DataFrame with OHLCV data
        volatility_window: On prend 24H pour la volatilité
        momentum_window: On prend 24H pour le momentum
        volatility_percentile: On prend le 80e percentile pour la volatilité basé sur le prix historique de l'asset
        momentum_percentile: Pareil que avant mais 90e

    Returns:
        (0: Bull, 1: Bear, 2: Consolidation)
    """ 

    returns = df['close'].pct_change()
    volatility = returns.rolling(window=volatility_window).std()
    momentum = df['close'].pct_change(periods=momentum_window)
    volatility_threshold = np.percentile(volatility.dropna(), volatility_percentile)
    momentum_threshold = np.percentile(momentum.abs().dropna(), momentum_percentile)

    regimes = pd.Series(2, index=df.index) 
    is_volatile = volatility >= volatility_threshold
    significant_momentum = momentum.abs() >= momentum_threshold
    bull_condition = (momentum > momentum_threshold) | ((momentum > 0) & is_volatile)
    regimes[bull_condition] = 0
    bear_condition = (momentum < -momentum_threshold) | ((momentum < 0) & is_volatile)
    regimes[bear_condition] = 1

    return regimes