import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Dict, Optional, Tuple

def stratified_sample_by_cluster(X: np.ndarray, labels: np.ndarray, sample_size: int = None, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выполняет стратифицированную выборку по кластерам.

    params:
        X: Данные для кластеризации
        labels: Предсказанные кластеры
        sample_size: Размер выборки

    returns:
        X_sampled, labels_sampled: Выборка данных и соответствующих меток кластеров
    """
    if sample_size is None:
        return X, labels
    
    np.random.seed(random_state)

    unique_labels = np.unique(labels)
    sampled_indices = []

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]        
        n_to_sample = min(sample_size, len(cluster_indices))
        sampled = np.random.choice(cluster_indices, size=n_to_sample, replace=False)
        sampled_indices.extend(sampled)

    sampled_indices = np.array(sampled_indices)
    X_sampled = X[sampled_indices]
    labels_sampled = labels[sampled_indices]

    return X_sampled, labels_sampled

def random_sample(X: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Возвращает случайную подвыборку без замены из массива.
    
    params:
        X: Входной массив
        n_samples: Количество элементов в выборке
    
    returns:
        X_sample: Случайная подвыборка
    """
    if n_samples > len(X):
        raise ValueError(f"Величина подвыборки ({n_samples}) не может быть больше размера массива ({len(X)})")
    
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]
    return X_sample


def calculate_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Расчет метрик качества кластеризации.

    params:
        X: Данные для кластеризации
        labels: Предсказанные кластеры

    returns:
        metrics: Словарь с метриками качества кластеризации
    """
    metrics = {
        "silhouette_score": silhouette_score(X, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X, labels),
        "davies_bouldin_score": davies_bouldin_score(X, labels)
    }
    return metrics

def silhouette_scoring(estimator, X):
    if hasattr(estimator, 'fit_predict'):
        labels = estimator.fit_predict(X)
    elif hasattr(estimator, 'predict'):
        labels = estimator.predict(X)
    else:
        raise ValueError(f"Estimator {type(estimator)} has neither predict nor fit_predict")

    if len(np.unique(labels)) < 2:
        return -1
    return silhouette_score(X, labels)

def calinski_harabasz_scoring(estimator, X):
    labels = estimator.predict(X)
    if len(np.unique(labels)) < 2:
        return -1
    return calinski_harabasz_score(X, labels)

def davies_bouldin_scoring(estimator, X):
    labels = estimator.predict(X)
    if len(np.unique(labels)) < 2:
        return -1
    return davies_bouldin_score(X, labels)