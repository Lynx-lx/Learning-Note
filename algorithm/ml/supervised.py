from __future__ import annotations

import numpy as np

from algorithm.common.base import Estimator


class LinearRegression(Estimator):
    """最小二乘线性回归（正规方程）。"""

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None

    def _design(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not self.fit_intercept:
            return x
        return np.column_stack([np.ones(len(x)), x])

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> LinearRegression:
        if y is None:
            raise ValueError("监督模型需要标签 y")
        a = self._design(x)
        y = np.asarray(y, dtype=float).ravel()
        self.coef_, *_ = np.linalg.lstsq(a, y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练")
        return self._design(x) @ self.coef_


class LogisticRegression(Estimator):
    """二分类逻辑回归（梯度下降）。"""

    def __init__(self, lr: float = 0.1, epochs: int = 500) -> None:
        self.lr = lr
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> LogisticRegression:
        if y is None:
            raise ValueError("监督模型需要标签 y")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = x.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            z = x @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            self.w -= self.lr * (x.T @ (p - y) / n)
            self.b -= self.lr * float(np.mean(p - y))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("模型尚未训练")
        z = np.asarray(x, dtype=float) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


class KNNClassifier(Estimator):
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> KNNClassifier:
        if y is None:
            raise ValueError("监督模型需要标签 y")
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x is None or self.y is None:
            raise RuntimeError("模型尚未训练")
        x = np.asarray(x, dtype=float)
        preds = []
        for row in x:
            dist = np.linalg.norm(self.x - row, axis=1)
            idx = np.argpartition(dist, self.k)[: self.k]
            votes, counts = np.unique(self.y[idx], return_counts=True)
            preds.append(votes[np.argmax(counts)])
        return np.asarray(preds)
