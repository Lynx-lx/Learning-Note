from __future__ import annotations

import numpy as np

from algorithm.slam import ICP2D


def main() -> None:
    rng = np.random.default_rng(3)
    dst = rng.normal(size=(40, 2))
    theta = np.deg2rad(20)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    trans = np.array([0.6, -0.4])
    src = dst @ rot.T + trans

    icp = ICP2D(max_iter=30)
    icp.fit(src, dst)
    aligned = icp.predict(src)
    rmse = float(np.sqrt(np.mean((aligned - dst) ** 2)))
    print("ICP RMSE:", round(rmse, 4))
    print("估计平移:", np.round(icp.t, 3))


if __name__ == "__main__":
    main()
