import networkx as nx
from community import community_louvain
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

class CryptoClusterer:
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        
    def prepare_features(self, df):
        """
        Prepare and normalize features for clustering.
        Handles infinite values and outliers.
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close']).diff()
        
        # Volatility features (with minimum value to avoid division by zero)
        for window in [24, 168]:  # 1 day, 1 week
            features[f'volatility_{window}h'] = features['returns'].rolling(window).std()
        
        # Volume features (with handling for zero values)
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(168).mean() + 1e-8)  # Add small constant
        
        # Momentum features
        for window in [24, 168]:  # 1 day, 1 week
            features[f'momentum_{window}h'] = df['close'].pct_change(window)
        
        # Handle infinite values and NaNs
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill, then backward fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # If any NaN values remain, fill with 0
        features = features.fillna(0)
        
        # Remove outliers (clip values beyond 3 standard deviations)
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            features[col] = features[col].clip(mean - 3*std, mean + 3*std)
        
        # Normalize features
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
        
        return pd.DataFrame(normalized, index=features.index, columns=features.columns)

    def louvain_clustering(self, features, threshold=0.5):
        """
        Apply Louvain method for community detection.
        """
        # Create correlation matrix
        corr_matrix = np.corrcoef(features.T)
        
        # Create network
        G = nx.Graph()
        n = corr_matrix.shape[0]
        
        # Add edges with weights based on correlations
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr_matrix[i,j]) > threshold:
                    G.add_edge(i, j, weight=abs(corr_matrix[i,j]))
        
        # Apply Louvain method
        communities = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(communities, G)
        
        return communities, modularity, G

    def dbscan_clustering(self, features, eps=0.5, min_samples=5):
        """
        Apply DBSCAN clustering.
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(features)
        
        # Calculate silhouette score if more than one cluster
        if len(set(labels)) > 1:
            silhouette = silhouette_score(features, labels)
        else:
            silhouette = 0
            
        return labels, silhouette

    def kmeans_clustering(self, features, n_clusters=3):
        """
        Apply K-means clustering.
        """
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clustering.fit_predict(features)
        
        silhouette = silhouette_score(features, labels)
        return labels, silhouette, clustering.cluster_centers_

    def get_optimal_clusters(self, features, max_clusters=10):
        """
        Find optimal number of clusters using silhouette score.
        """
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            labels, silhouette, _ = self.kmeans_clustering(features, n_clusters)
            silhouette_scores.append(silhouette)
        
        optimal_clusters = np.argmax(silhouette_scores) + 2
        return optimal_clusters, silhouette_scores 