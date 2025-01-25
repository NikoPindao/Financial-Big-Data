from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform

class ClusteringUtils:
    @staticmethod
    def preprocess_features(features: pd.DataFrame) -> pd.DataFrame:
        """Standardize features and handle missing values"""
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Standardize
        scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
        return scaled_features
    
    @staticmethod
    def calculate_optimal_dbscan_params(feature_array: np.ndarray) -> Tuple[float, int]:
        """Calculate optimal DBSCAN parameters using distance distribution"""
        # Calculate pairwise distances
        distances = pdist(feature_array)
        dist_matrix = squareform(distances)
        
        # Calculate eps using distance distribution
        k = min(15, len(feature_array) - 1)
        knn_distances = np.sort(dist_matrix, axis=0)[1:k+1]
        eps = np.percentile(knn_distances.mean(axis=0), 75)
        
        # Calculate min_samples based on dimensionality
        min_samples = max(3, feature_array.shape[1])
        
        return eps, min_samples
    
    @staticmethod
    def evaluate_clustering(feature_array: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics"""
        metrics = {}
        
        # Handle noise points for DBSCAN
        if -1 in labels:
            valid_mask = labels != -1
            if np.sum(valid_mask) > 1:
                metrics['silhouette'] = silhouette_score(
                    feature_array[valid_mask],
                    labels[valid_mask]
                )
                metrics['calinski'] = calinski_harabasz_score(
                    feature_array[valid_mask],
                    labels[valid_mask]
                )
        else:
            metrics['silhouette'] = silhouette_score(feature_array, labels)
            metrics['calinski'] = calinski_harabasz_score(feature_array, labels)
        
        return metrics 