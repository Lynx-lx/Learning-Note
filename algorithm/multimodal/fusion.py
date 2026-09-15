from __future__ import annotations

import numpy as np

from algorithm.common.base import Transform


class ConcatFusion(Transform):
    """早期融合：将 (模态A, 模态B) 在特征维拼接。"""

    def __call__(self, data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        a, b = data
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]
        return np.hstack([a, b])
