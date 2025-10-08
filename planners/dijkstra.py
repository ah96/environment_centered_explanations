#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dijkstra planner for grid maps (4- or 8-connected).
- Uniform edge relaxation, no heuristic (A* with h=0).
- Costs: 1 cardinal, sqrt(2) diagonals (if 8-connected).
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import heapq
import math
import numpy as np


class DijkstraPlanner:
    def __init__(self, connectivity: int = 8):
        assert connectivity in (4, 8)
        self.conn = connectivity
        if connectivity == 8:
            self.deltas = np.array([
                (-1, -1), (-1, 0), (-1, +1),
                ( 0, -1),          ( 0, +1),
                (+1, -1), (+1, 0), (+1, +1)
            ], dtype=np.int8)
            self.costs = np.array([math.sqrt(2), 1, math.sqrt(2),
                                   1,               1,
                                   math.sqrt(2), 1, math.sqrt(2)], dtype=np.float32)
        else:
            self.deltas = np.array([(-1,0), (1,0), (0,-1), (0,1)], dtype=np.int8)
            self.costs = np.ones(4, dtype=np.float32)

    @staticmethod
    def _reconstruct(par_r: np.ndarray, par_c: np.ndarray,
                     start: Tuple[int,int], goal: Tuple[int,int]):
        path = []
        r, c = goal
        while (r, c) != start:
            path.append((int(r), int(c)))
            pr, pc = int(par_r[r, c]), int(par_c[r, c])
            if pr == -1 and pc == -1:  # No parent found
                return None
            r, c = pr, pc
        path.append(start)
        path.reverse()
        return path

    def plan(self, grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Dict:
        H, W = grid.shape
        sr, sc = start; gr, gc = goal
        
        # Validate bounds
        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            return {'success': False, 'path': None}
        
        if grid[sr, sc] or grid[gr, gc]:
            return {'success': False, 'path': None}
        
        # Handle trivial case
        if start == goal:
            return {'success': True, 'path': [start]}

        dist = np.full((H, W), np.inf, dtype=np.float32)
        par_r = np.full((H, W), -1, dtype=np.int32)
        par_c = np.full((H, W), -1, dtype=np.int32)
        visited = np.zeros((H, W), dtype=bool)

        dist[sr, sc] = 0.0
        pq: List[Tuple[float, int, int]] = [(0.0, sr, sc)]

        while pq:
            d, r, c = heapq.heappop(pq)
            if visited[r, c]:
                continue
            visited[r, c] = True
            if (r, c) == (gr, gc):
                path = self._reconstruct(par_r, par_c, start, goal)
                return {'success': True, 'path': path}
            for k, (dr, dc) in enumerate(self.deltas):
                nr, nc = r + int(dr), c + int(dc)
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if grid[nr, nc] or visited[nr, nc]:
                    continue
                nd = d + self.costs[k]
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    par_r[nr, nc] = r
                    par_c[nr, nc] = c
                    heapq.heappush(pq, (nd, nr, nc))

        return {'success': False, 'path': None}