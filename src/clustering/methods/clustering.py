from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import networkx as nx
from community import community_louvain

class BaseClusterer(ABC):
    """Abstract base class for clustering methods"""
    
    @abstractmethod
    def fit(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Fit clustering model and return labels with metrics"""
        pass
    
    def preprocess_features(self, features: pd.DataFrame) -> np.ndarray:
        """Standardize features"""
        scaler = StandardScaler()
        return scaler.fit_transform(features)

class KMeansClusterer(BaseClusterer):
    """K-means clustering for regime detection"""
    
    def __init__(self, n_clusters: int = 5, n_init: int = 10):
        self.n_clusters = n_clusters
        self.n_init = n_init
    
    def fit(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        scaled_features = self.preprocess_features(features)
        
        # Fit K-means with explicit n_init
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42,
            n_init=self.n_init
        )
        labels = kmeans.fit_predict(scaled_features)
        
        # Calculate metrics
        silhouette = silhouette_score(scaled_features, labels)
        inertia = kmeans.inertia_
        
        # Create results
        clusters = pd.DataFrame({
            'cluster': labels,
            'silhouette': silhouette
        }, index=features.index)
        
        metrics = {
            'silhouette_score': silhouette,
            'inertia': inertia,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        return clusters, metrics

class GraphClusterer(BaseClusterer):
    """Graph-based clustering using Louvain method"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def fit(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        scaled_features = self.preprocess_features(features)
        
        # Create correlation network
        corr_matrix = np.corrcoef(scaled_features.T)
        G = nx.Graph()
        
        # Add edges based on correlation
        n = corr_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr_matrix[i,j]) > self.threshold:
                    G.add_edge(i, j, weight=abs(corr_matrix[i,j]))
        
        # Apply Louvain clustering
        communities = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(communities, G)
        
        # Create results
        clusters = pd.DataFrame({
            'cluster': pd.Series(communities),
            'modularity': modularity
        }, index=features.index)
        
        metrics = {
            'modularity': modularity,
            'n_communities': len(set(communities.values())),
            'graph_density': nx.density(G)
        }
        
        return clusters, metrics 