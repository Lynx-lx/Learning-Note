from __future__ import annotations

import numpy as np

from algorithm.common.base import Estimator


class MatrixFactorization(Estimator):
    """评分矩阵分解。x 为 [user_id, item_id]，y 为评分。"""

    def __init__(self, n_factors: int = 8, lr: float = 0.02, epochs: int = 80, n_users: int = 0, n_items: int = 0) -> None:
        self.n_factors = n_factors
        self.lr = lr
        self.epochs = epochs
        self.n_users = n_users
        self.n_items = n_items
        self.user_f: np.ndarray | None = None
        self.item_f: np.ndarray | None = None
        self.user_b: np.ndarray | None = None
        self.item_b: np.ndarray | None = None
        self.global_mean = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> MatrixFactorization:
        if y is None:
            raise ValueError("协同过滤需要评分 y")
        x = np.asarray(x, dtype=int)
        y = np.asarray(y, dtype=float).ravel()
        users, items = x[:, 0], x[:, 1]
        self.n_users = max(self.n_users, int(users.max()) + 1)
        self.n_items = max(self.n_items, int(items.max()) + 1)
        rng = np.random.default_rng(0)
        self.user_f = 0.1 * rng.normal(size=(self.n_users, self.n_factors))
        self.item_f = 0.1 * rng.normal(size=(self.n_items, self.n_factors))
        self.user_b = np.zeros(self.n_users)
        self.item_b = np.zeros(self.n_items)
        self.global_mean = float(y.mean())
        for _ in range(self.epochs):
            for (u, i), r in zip(x, y):
                pred = self._score(int(u), int(i))
                err = r - pred
                self.user_b[u] += self.lr * err
                self.item_b[i] += self.lr * err
                uf = self.user_f[u].copy()
                self.user_f[u] += self.lr * err * self.item_f[i]
                self.item_f[i] += self.lr * err * uf
        return self

    def _score(self, u: int, i: int) -> float:
        return float(self.global_mean + self.user_b[u] + self.item_b[i] + self.user_f[u] @ self.item_f[i])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.user_f is None:
            raise RuntimeError("模型尚未训练")
        x = np.asarray(x, dtype=int)
        return np.array([self._score(int(u), int(i)) for u, i in x])
