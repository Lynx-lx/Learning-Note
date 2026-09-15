from __future__ import annotations

import numpy as np

from algorithm.control import GridAStar, PIDController


def simulate_plant(pid: PIDController, start: float = 0.0, steps: int = 40) -> np.ndarray:
    pos, vel = start, 0.0
    traj = []
    pid.reset()
    for _ in range(steps):
        u = float(pid.predict(np.array([pos]))[0])
        vel = 0.8 * vel + 0.2 * u
        pos = pos + 0.1 * vel
        traj.append(pos)
    return np.asarray(traj)


def main() -> None:
    grid = np.zeros((8, 8), dtype=int)
    grid[1:7, 3] = 1
    grid[4, 3] = 0
    planner = GridAStar()
    planner.fit(grid)
    path = planner.predict(np.array([0, 0, 7, 7]))
    print("A* 路径长度:", len(path), "终点:", path[-1].tolist())

    pid = PIDController()
    pid.fit(np.zeros(1), y=np.array([1.0]))
    traj = simulate_plant(pid)
    print("PID 末位置:", round(float(traj[-1]), 3), "目标: 1.0")


if __name__ == "__main__":
    main()
