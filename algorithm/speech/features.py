from __future__ import annotations

import numpy as np

from algorithm.common.base import Transform


class WaveformNormalize(Transform):
    """去掉直流分量，保留能量尺度，便于后续用能量区分有声/无声。"""

    def __call__(self, waves: np.ndarray) -> np.ndarray:
        waves = np.asarray(waves, dtype=float)
        return waves - waves.mean(axis=-1, keepdims=True)


class FrameFeature(Transform):
    """从波形提取能量、过零率、标准差，便于接入分类器。"""

    def __call__(self, waves: np.ndarray) -> np.ndarray:
        waves = np.asarray(waves, dtype=float)
        energy = np.mean(waves**2, axis=1, keepdims=True)
        zcr = np.mean(np.abs(np.diff(np.sign(waves), axis=1)), axis=1, keepdims=True)
        std = np.std(waves, axis=1, keepdims=True)
        return np.hstack([energy, zcr, std])
