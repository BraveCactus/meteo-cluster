import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, Optional
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .base import BaseClusterer
from sklearn.cluster import DBSCAN


class DBSCANClusterer(BaseClusterer):
    """DBSCAN clustering implementation."""
    
    def __init__(self, eps: float = 1.5, min_samples: int = 8, random_state: int = 42):
        super().__init__(name="dbscan")
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.n_clusters_ = None
        self.noise_count_ = None
    
    def _create_model(self, **kwargs):
        return DBSCAN(eps=self.eps, 
                      min_samples=self.min_samples)
  
    def plot_k_distance_graph(self, X: np.ndarray, k: int = None) -> None:
        """
        Визуализирует график k-расстояний для выбора eps.
        
        params:
            X: Данные для кластеризации
            k: Количество ближайших соседей
        """        
        
        if k is None:
            k = self.min_samples
        
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)     
        k_distances = np.sort(distances[:, -1])        
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(k_distances)), k_distances, 'b-', alpha=0.7)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'Distance to {k}-th nearest neighbor')
        plt.title(f'k-Distance Graph (k={k})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

