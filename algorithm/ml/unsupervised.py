from __future__ import annotations

import numpy as np

from algorithm.common.base import Estimator


class KMeans(Estimator):
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, random_state: int = 0) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> KMeans:
        rng = np.random.default_rng(self.random_state)
        x = np.asarray(x, dtype=float)
        idx = rng.choice(len(x), size=self.n_clusters, replace=False)
        centers = x[idx].copy()
        for _ in range(self.max_iter):
            labels = self._assign(x, centers)
            new_centers = np.stack(
                [x[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in range(self.n_clusters)]
            )
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        self.centers_ = centers
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("模型尚未训练")
        return self._assign(np.asarray(x, dtype=float), self.centers_)

    @staticmethod
    def _assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return dist.argmin(axis=1)
