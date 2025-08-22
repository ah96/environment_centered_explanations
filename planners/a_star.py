#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A* path planner for grid maps (4- or 8-connected).
- Obstacles are True in `grid`; free space is False.
- Heuristic: Manhattan (4-connected) or Octile (8-connected).
- Edge costs: 1 for cardinal moves; sqrt(2) for diagonals.

Returns {'success': bool, 'path': list[(r,c)] or None}.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import heapq
import math
import numpy as np


class AStarPlanner:
    def __init__(self, connectivity: int = 8):
        assert connectivity in (4, 8)
        self.conn = connectivity
        # Neighbor deltas and step costs
        if connectivity == 8:
            self.deltas = np.array([
                (-1, -1), (-1, 0), (-1, +1),
                ( 0, -1),          ( 0, +1),
                (+1, -1), (+1, 0), (+1, +1)
            ], dtype=np.int8)
            self.costs = np.array([math.sqrt(2), 1, math.sqrt(2),
                                   1,               1,
                                   math.sqrt(2), 1, math.sqrt(2)], dtype=np.float32)
        else:  # 4-connected
            self.deltas = np.array([(-1,0), (1,0), (0,-1), (0,1)], dtype=np.int8)
            self.costs = np.ones(4, dtype=np.float32)

    def _heuristic(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        dr = abs(a[0] - b[0]); dc = abs(a[1] - b[1])
        if self.conn == 4:
            return float(dr + dc)  # Manhattan
        # Octile
        dmin, dmax = (dr, dc) if dr < dc else (dc, dr)
        return float((math.sqrt(2) - 1) * dmin + dmax)

    @staticmethod
    def _reconstruct(par_r: np.ndarray, par_c: np.ndarray,
                     start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        if par_r[goal] == -1 and goal != start:
            return None
        path: List[Tuple[int,int]] = []
        r, c = goal
        while (r, c) != start:
            path.append((int(r), int(c)))
            pr, pc = par_r[r, c], par_c[r, c]
            if pr == -1:  # no parent (shouldn't happen if reachable)
                return None
            r, c = int(pr), int(pc)
        path.append(start)
        path.reverse()
        return path

    def plan(self, grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Dict:
        H, W = grid.shape
        sr, sc = start; gr, gc = goal
        if grid[sr, sc] or grid[gr, gc]:
            return {'success': False, 'path': None}

        g = np.full((H, W), np.inf, dtype=np.float32)
        f = np.full((H, W), np.inf, dtype=np.float32)
        in_open = np.zeros((H, W), dtype=bool)
        closed = np.zeros((H, W), dtype=bool)
        par_r = np.full((H, W), -1, dtype=np.int32)
        par_c = np.full((H, W), -1, dtype=np.int32)

        g[sr, sc] = 0.0
        f[sr, sc] = self._heuristic(start, goal)

        pq: List[Tuple[float, int, int]] = []
        heapq.heappush(pq, (f[sr, sc], sr, sc))
        in_open[sr, sc] = True

        while pq:
            _, r, c = heapq.heappop(pq)
            if closed[r, c]:
                continue
            closed[r, c] = True
            if (r, c) == (gr, gc):
                path = self._reconstruct(par_r, par_c, start, goal)
                return {'success': True, 'path': path}

            for k, (dr, dc) in enumerate(self.deltas):
                nr, nc = r + int(dr), c + int(dc)
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if grid[nr, nc] or closed[nr, nc]:
                    continue
                tentative = g[r, c] + self.costs[k]
                if tentative < g[nr, nc]:
                    g[nr, nc] = tentative
                    par_r[nr, nc] = r
                    par_c[nr, nc] = c
                    f[nr, nc] = tentative + self._heuristic((nr, nc), goal)
                    heapq.heappush(pq, (f[nr, nc], nr, nc))
                    in_open[nr, nc] = True

        return {'success': False, 'path': None}
