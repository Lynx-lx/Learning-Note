from __future__ import annotations

import numpy as np

from algorithm.common.base import Transform


class CountVectorizer(Transform):
    """空格分词词袋。输入文本列表或一维 object 数组，输出稠密计数矩阵。"""

    def __init__(self) -> None:
        self.vocab: dict[str, int] | None = None

    def fit(self, texts: np.ndarray | list[str]) -> CountVectorizer:
        tokens: list[str] = []
        for t in texts:
            tokens.extend(str(t).lower().split())
        self.vocab = {w: i for i, w in enumerate(sorted(set(tokens)))}
        return self

    def __call__(self, texts: np.ndarray | list[str]) -> np.ndarray:
        texts = list(texts)
        if self.vocab is None:
            self.fit(texts)
        mat = np.zeros((len(texts), len(self.vocab)), dtype=float)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                j = self.vocab.get(w)
                if j is not None:
                    mat[i, j] += 1.0
        return mat
