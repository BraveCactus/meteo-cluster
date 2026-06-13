from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import Any, Dict, Optional

class BaseClusterer(ABC, BaseEstimator, ClusterMixin):
    """Базовыйй класс для всех кластеризаторов"""

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.is_fitted = False

    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """"Создает экземпляр модели с заданными параметрами"""
        pass

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Обучает модель на данных X"""

        self.model = self._create_model(**kwargs)        
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True
        return self    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает кластеры для данных X"""

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict(X)
    
    def save(self, file_path: Path) -> None:
        """Сохраняет модель"""

        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, file_path)

    def load(self, file_path: Path) -> None:
        """Загружает модель"""

        if not file_path.exists():
            raise ValueError("File does not exist.")
        self.model = joblib.load(file_path)
        self.is_fitted = True


        
    