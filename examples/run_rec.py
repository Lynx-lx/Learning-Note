from __future__ import annotations

import numpy as np

from algorithm.common.metrics import regression_report
from algorithm.rec import MatrixFactorization


def main() -> None:
    rng = np.random.default_rng(2)
    n_users, n_items = 12, 10
    true_u = rng.normal(size=(n_users, 3))
    true_i = rng.normal(size=(n_items, 3))
    pairs = np.array([[u, i] for u in range(n_users) for i in range(n_items) if rng.random() < 0.5])
    ratings = np.array([true_u[u] @ true_i[i] for u, i in pairs])
    ratings = ratings + 0.05 * rng.normal(size=len(ratings))

    model = MatrixFactorization(n_factors=3, epochs=60)
    model.fit(pairs, ratings)
    pred = model.predict(pairs)
    print("矩阵分解:", regression_report(ratings, pred))


if __name__ == "__main__":
    main()
