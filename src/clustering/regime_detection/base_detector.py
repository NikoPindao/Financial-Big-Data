from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

@dataclass
class BaseThresholds:
    """Base thresholds for regime classification"""
    volatility_threshold: float
    return_threshold: float
    volume_threshold: float
    trend_threshold: float
    
class BaseRegimeDetector(ABC):
    """Abstract base class for regime detection"""
    
    def __init__(self, 
                 window_size: int,
                 overlap: int,
                 thresholds: BaseThresholds):
        self.window_size = window_size
        self.overlap = overlap
        self.thresholds = thresholds
    
    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection"""
        pass
    
    @abstractmethod
    def detect_regime(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect market regime"""
        pass 