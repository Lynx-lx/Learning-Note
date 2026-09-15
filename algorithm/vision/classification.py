from __future__ import annotations

import numpy as np

from algorithm.common.base import Estimator
from algorithm.vision.preprocess import flatten_image


class SoftmaxClassifier(Estimator):
    """图像分类最小实现：将图像展平后做线性 Softmax。"""

    def __init__(self, n_classes: int, lr: float = 0.05, epochs: int = 200) -> None:
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 4:
            return x.reshape(len(x), -1)
        if x.ndim == 3:
            return flatten_image(x)[None, :]
        return x

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> SoftmaxClassifier:
        if y is None:
            raise ValueError("分类模型需要标签 y")
        x = self._prepare(x)
        y = np.asarray(y, dtype=int).ravel()
        n, d = x.shape
        self.w = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        eye = np.eye(self.n_classes)
        for _ in range(self.epochs):
            logits = x @ self.w + self.b
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            prob = exp / exp.sum(axis=1, keepdims=True)
            onehot = eye[y]
            grad_w = x.T @ (prob - onehot) / n
            grad_b = (prob - onehot).mean(axis=0)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None or self.b is None:
            raise RuntimeError("模型尚未训练")
        logits = self._prepare(x) @ self.w + self.b
        return logits.argmax(axis=1)
