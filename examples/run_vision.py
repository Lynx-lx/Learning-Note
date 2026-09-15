from __future__ import annotations

import numpy as np

from algorithm.common.metrics import classification_report
from algorithm.common.pipeline import Pipeline
from algorithm.vision import ImageNormalize, SoftmaxClassifier


def main() -> None:
    rng = np.random.default_rng(1)
    n, h, w, c = 40, 8, 8, 3
    images = rng.random((n, h, w, c)).astype(np.float32)
    labels = (images.mean(axis=(1, 2, 3)) > 0.5).astype(int)

    pipe = Pipeline(steps=[ImageNormalize()], estimator=SoftmaxClassifier(n_classes=2, epochs=80))
    pipe.fit(images, labels)
    pred = pipe.predict(images)
    print("图像分类:", classification_report(labels, pred))


if __name__ == "__main__":
    main()
