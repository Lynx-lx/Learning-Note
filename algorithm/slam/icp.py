from __future__ import annotations

import numpy as np

from algorithm.common.base import Estimator


class ICP2D(Estimator):
    """二维 ICP：fit(源点云, 目标点云) 估计刚体变换，predict 把新点变换到目标系。"""

    def __init__(self, max_iter: int = 20) -> None:
        self.max_iter = max_iter
        self.R = np.eye(2)
        self.t = np.zeros(2)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> ICP2D:
        if y is None:
            raise ValueError("ICP 需要目标点云 y")
        src = np.asarray(x, dtype=float)
        dst = np.asarray(y, dtype=float)
        R, t = np.eye(2), np.zeros(2)
        cur = src.copy()
        for _ in range(self.max_iter):
            matched = self._nearest(cur, dst)
            R_step, t_step = self._rigid(cur, matched)
            cur = cur @ R_step.T + t_step
            R = R_step @ R
            t = R_step @ t + t_step
        self.R, self.t = R, t
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        src = np.asarray(x, dtype=float)
        return src @ self.R.T + self.t

    @staticmethod
    def _nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        dist = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
        return dst[dist.argmin(axis=1)]

    @staticmethod
    def _rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cs, cd = src.mean(axis=0), dst.mean(axis=0)
        h = (src - cs).T @ (dst - cd)
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1] *= -1
            r = vt.T @ u.T
        t = cd - r @ cs
        return r, t
