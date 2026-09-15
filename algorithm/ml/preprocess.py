from __future__ import annotations

import numpy as np

from algorithm.common.base import Transform


class StandardScaler(Transform):
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.mean_ is None or self.std_ is None:
            self.fit(x)
        return (x - self.mean_) / self.std_
