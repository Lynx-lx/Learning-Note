from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from algorithm.common.base import Estimator, Transform


@dataclass
class Pipeline:
    """预处理 + 估计器的最小流水线。"""

    steps: list[Transform] = field(default_factory=list)
    estimator: Estimator | None = None

    def transform(self, x: Any) -> Any:
        for step in self.steps:
            x = step(x)
        return x

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Pipeline:
        if self.estimator is None:
            raise ValueError("pipeline 未设置 estimator")
        x = self.transform(x)
        self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("pipeline 未设置 estimator")
        x = self.transform(x)
        return self.estimator.predict(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], dict]) -> dict:
        pred = self.predict(x)
        return metric(y, pred)
