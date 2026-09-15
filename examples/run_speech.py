from __future__ import annotations

import numpy as np

from algorithm.common.metrics import classification_report
from algorithm.common.pipeline import Pipeline
from algorithm.ml import LogisticRegression
from algorithm.speech import FrameFeature, WaveformNormalize


def main() -> None:
    rng = np.random.default_rng(4)
    n, t = 60, 200
    quiet = 0.05 * rng.normal(size=(n // 2, t))
    voiced = rng.normal(size=(n // 2, t))
    waves = np.vstack([quiet, voiced])
    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    pipe = Pipeline(steps=[WaveformNormalize(), FrameFeature()], estimator=LogisticRegression(epochs=300))
    pipe.fit(waves, labels)
    pred = pipe.predict(waves)
    print("语音有无声分类:", classification_report(labels, pred))


if __name__ == "__main__":
    main()
