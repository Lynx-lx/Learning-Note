from __future__ import annotations

import numpy as np

from algorithm.common.metrics import classification_report, regression_report
from algorithm.common.pipeline import Pipeline
from algorithm.ml import KMeans, LinearRegression, LogisticRegression, StandardScaler


def main() -> None:
    rng = np.random.default_rng(0)

    x_reg = rng.normal(size=(80, 2))
    y_reg = 3 * x_reg[:, 0] - 2 * x_reg[:, 1] + 0.1 * rng.normal(size=80)
    scaler = StandardScaler()
    scaler.fit(x_reg)
    pipe = Pipeline(steps=[scaler], estimator=LinearRegression())
    pipe.fit(x_reg, y_reg)
    print("线性回归:", pipe.evaluate(x_reg, y_reg, regression_report))

    x_clf = rng.normal(size=(100, 2))
    y_clf = (x_clf[:, 0] + x_clf[:, 1] > 0).astype(int)
    logreg = LogisticRegression()
    logreg.fit(x_clf, y_clf)
    print("逻辑回归:", classification_report(y_clf, logreg.predict(x_clf)))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(x_clf)
    print("KMeans 簇大小:", np.bincount(kmeans.predict(x_clf)))


if __name__ == "__main__":
    main()
