#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breadth-First Search planner (unweighted shortest hops).
- Works on 4- or 8-connected grids.
- Treats all moves as unit cost (diagonals count as 1 if enabled).
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from collections import deque
import numpy as np


class BFSPlanner:
    def __init__(self, connectivity: int = 8):
        assert connectivity in (4, 8)
        self.conn = connectivity
        if connectivity == 8:
            self.deltas = np.array([
                (-1, -1), (-1, 0), (-1, +1),
                ( 0, -1),          ( 0, +1),
                (+1, -1), (+1, 0), (+1, +1)
            ], dtype=np.int8)
        else:
            self.deltas = np.array([(-1,0), (1,0), (0,-1), (0,1)], dtype=np.int8)

    @staticmethod
    def _reconstruct(par_r: np.ndarray, par_c: np.ndarray,
                     start: Tuple[int,int], goal: Tuple[int,int]):
        if par_r[goal] == -1 and goal != start:
            return None
        path = []
        r, c = goal
        while (r, c) != start:
            path.append((int(r), int(c)))
            pr, pc = par_r[r, c], par_c[r, c]
            if pr == -1: return None
            r, c = int(pr), int(pc)
        path.append(start)
        path.reverse()
        return path

    def plan(self, grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Dict:
        H, W = grid.shape
        sr, sc = start; gr, gc = goal
        if grid[sr, sc] or grid[gr, gc]:
            return {'success': False, 'path': None}
        if start == goal:
            return {'success': True, 'path': [start]}

        visited = np.zeros((H, W), dtype=bool)
        par_r = np.full((H, W), -1, dtype=np.int32)
        par_c = np.full((H, W), -1, dtype=np.int32)

        dq = deque()
        dq.append((sr, sc))
        visited[sr, sc] = True

        while dq:
            r, c = dq.popleft()
            if (r, c) == (gr, gc):
                path = self._reconstruct(par_r, par_c, start, goal)
                return {'success': True, 'path': path}
            for dr, dc in self.deltas:
                nr, nc = r + int(dr), c + int(dc)
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if visited[nr, nc] or grid[nr, nc]:
                    continue
                visited[nr, nc] = True
                par_r[nr, nc] = r
                par_c[nr, nc] = c
                dq.append((nr, nc))

        return {'success': False, 'path': None}
