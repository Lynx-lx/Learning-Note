from __future__ import annotations

import numpy as np

from algorithm.common.metrics import classification_report
from algorithm.common.pipeline import Pipeline
from algorithm.ml import LogisticRegression
from algorithm.multimodal import ConcatFusion


def main() -> None:
    rng = np.random.default_rng(5)
    n = 80
    image_feat = rng.normal(size=(n, 6))
    text_feat = rng.normal(size=(n, 4))
    labels = ((image_feat[:, 0] + text_feat[:, 0]) > 0).astype(int)

    pipe = Pipeline(steps=[ConcatFusion()], estimator=LogisticRegression(epochs=400))
    pipe.fit((image_feat, text_feat), labels)
    pred = pipe.predict((image_feat, text_feat))
    print("多模态早期融合分类:", classification_report(labels, pred))


if __name__ == "__main__":
    main()
