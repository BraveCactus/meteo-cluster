from sklearn.cluster import KMeans
from typing import Dict, Any, Literal
from pathlib import Path

from .base import BaseClusterer

class KMeansClusterer(BaseClusterer):
    """Кластеризатор на основе алгоритма KMeans"""

    def __init__(self, 
                 n_clusters: int = 8, 
                 init: str = 'k-means++', 
                 n_init: int | str = 'auto', 
                 max_iter: int = 300, 
                 tol: float = 0.0001, 
                 verbose: int = 0, 
                 algorithm: Literal["lloyd", "elkan"] = "lloyd",
                 random_state: int = 42):
        super().__init__(name="KMeans", random_state=random_state)
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose        
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, 
                            init=self.init, 
                            n_init=self.n_init, 
                            max_iter=self.max_iter, 
                            tol=self.tol, 
                            verbose=self.verbose, 
                            random_state=self.random_state, 
                            algorithm=self.algorithm)

    def _create_model(self, **kwargs) -> KMeans:
        """Создает экземпляр модели KMeans с заданными параметрами"""
        return KMeans(n_clusters=self.n_clusters, 
                      init=self.init, 
                      n_init=self.n_init, 
                      max_iter=self.max_iter, 
                      tol=self.tol, 
                      verbose=self.verbose, 
                      random_state=self.random_state, 
                      algorithm=self.algorithm)
    
    def get_params(self) -> Dict[str, Any]:
        """Возвращает параметры модели KMeans"""
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "verbose": self.verbose,
            "algorithm": self.algorithm,
            "random_state": self.random_state
        }
    