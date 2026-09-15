from __future__ import annotations

import numpy as np
from PIL import Image

from algorithm.common.base import Transform


def load_image(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0


class Resize(Transform):
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        arr = np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.resize(self.size)
        return np.asarray(img, dtype=np.float32) / 255.0


class ImageNormalize(Transform):
    def __init__(self, mean: tuple[float, float, float] = (0.485, 0.456, 0.406), std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        return (image - self.mean) / self.std


def flatten_image(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=float).reshape(-1)
