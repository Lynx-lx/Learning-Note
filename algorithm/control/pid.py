from __future__ import annotations

import numpy as np

from algorithm.common.base import Estimator


class PIDController(Estimator):
    """一维 PID。fit 记录目标值，predict 按当前观测输出控制量。"""

    def __init__(self, kp: float = 1.2, ki: float = 0.15, kd: float = 0.05, dt: float = 0.1) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.setpoint = 0.0
        self.integral = 0.0
        self.prev_err = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> PIDController:
        if y is None:
            raise ValueError("PID 需要目标序列 y")
        self.setpoint = float(np.mean(np.asarray(y, dtype=float)))
        self.integral = 0.0
        self.prev_err = 0.0
        return self

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_err = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        outs = []
        for val in np.asarray(x, dtype=float).ravel():
            err = self.setpoint - val
            self.integral += err * self.dt
            deriv = (err - self.prev_err) / self.dt
            outs.append(self.kp * err + self.ki * self.integral + self.kd * deriv)
            self.prev_err = err
        return np.asarray(outs)
