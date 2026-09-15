from __future__ import annotations

import heapq

import numpy as np

from algorithm.common.base import Estimator


class GridAStar(Estimator):
    """栅格 A*。fit 存占用栅格（1 为障碍），predict 输入 [start_r, start_c, goal_r, goal_c]。"""

    def __init__(self) -> None:
        self.grid: np.ndarray | None = None
        self.last_path: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> GridAStar:
        self.grid = np.asarray(x, dtype=int)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError("尚未载入地图")
        query = np.asarray(x, dtype=int).ravel()
        start = (int(query[0]), int(query[1]))
        goal = (int(query[2]), int(query[3]))
        path = self._astar(start, goal)
        self.last_path = path
        if path.size == 0:
            return np.array([[*start]], dtype=int)
        return path

    def _astar(self, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
        grid = self.grid
        h, w = grid.shape
        if not self._free(start, h, w) or not self._free(goal, h, w):
            return np.empty((0, 2), dtype=int)

        def heur(p: tuple[int, int]) -> int:
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        open_heap: list[tuple[int, int, tuple[int, int]]] = [(heur(start), 0, start)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        gscore = {start: 0}
        seen: set[tuple[int, int]] = set()
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))

        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur in seen:
                continue
            seen.add(cur)
            if cur == goal:
                return self._rebuild(came, cur)
            for dr, dc in deltas:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not self._free(nxt, h, w):
                    continue
                ng = g + 1
                if ng < gscore.get(nxt, 1_000_000):
                    gscore[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_heap, (ng + heur(nxt), ng, nxt))
        return np.empty((0, 2), dtype=int)

    def _free(self, p: tuple[int, int], h: int, w: int) -> bool:
        r, c = p
        if r < 0 or c < 0 or r >= h or c >= w:
            return False
        return int(self.grid[r, c]) == 0

    @staticmethod
    def _rebuild(came: dict[tuple[int, int], tuple[int, int]], cur: tuple[int, int]) -> np.ndarray:
        path = [cur]
        while cur in came:
            cur = came[cur]
            path.append(cur)
        path.reverse()
        return np.asarray(path, dtype=int)
