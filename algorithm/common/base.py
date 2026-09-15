from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Estimator(ABC):
    """统一估计器接口，各算法方向均可复用。"""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Estimator:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Transform(ABC):
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        raise NotImplementedError
