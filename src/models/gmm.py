from sklearn.mixture import GaussianMixture
from typing import Dict, Any

from .base import BaseClusterer

class GMMClusterer(BaseClusterer):
    """Кластеризатор на основе алгоритма Gaussian Mixture Model"""

    def __init__(self, 
                 n_components: int = 1, 
                 covariance_type: str = 'full', 
                 tol: float = 1e-3, 
                 reg_covar: float = 1e-6, 
                 max_iter: int = 100, 
                 n_init: int = 1, 
                 init_params: str = 'kmeans', 
                 random_state: int = 42):
        super().__init__(name="GMM", random_state=random_state)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.model = GaussianMixture(n_components=self.n_components, 
                                     covariance_type=self.covariance_type, 
                                     tol=self.tol, 
                                     reg_covar=self.reg_covar, 
                                     max_iter=self.max_iter, 
                                     n_init=self.n_init, 
                                     init_params=self.init_params, 
                                     random_state=self.random_state)

    def _create_model(self, **kwargs) -> GaussianMixture:
        """Создает экземпляр модели GaussianMixture с заданными параметрами"""
        return GaussianMixture(n_components=self.n_components, 
                               covariance_type=self.covariance_type, 
                               tol=self.tol, 
                               reg_covar=self.reg_covar, 
                               max_iter=self.max_iter, 
                               n_init=self.n_init, 
                               init_params=self.init_params, 
                               random_state=self.random_state)
    
    # def get_params(self) -> Dict[str, Any]:
    #     """Возвращает параметры модели GaussianMixture"""
    #     return {
    #         "n_components": self.n_components,
    #         "covariance_type": self.covariance_type,
    #         "tol": self.tol,
    #         "reg_covar": self.reg_covar,
    #         "max_iter": self.max_iter,
    #         "n_init": self.n_init,
    #         "init_params": self.init_params,
    #         "random_state": self.random_state
    #     }
    
    def update_params(self, **kwargs) -> None:
        """Обновляет параметры модели GaussianMixture"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model = self._create_model(**self.get_params())