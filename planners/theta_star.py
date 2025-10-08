#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theta* (any-angle A* with line-of-sight shortcuts).
- Uses A* search but attempts to connect successors directly to the parent's parent
  if line-of-sight is clear, producing shorter, smoother paths.
- 8-connected recommended for any-angle behavior.

LoS test uses a supercover Bresenham line that ensures all crossed cells are free.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import heapq
import math
import numpy as np


def _los_free(grid, a, b):
    """Check line-of-sight between two points using Bresenham."""
    r0, c0 = a
    r1, c1 = b
    
    # Check if endpoints are the same
    if (r0, c0) == (r1, c1):
        return not grid[r0, c0]
    
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    r, c = r0, c0
    
    if dr >= dc:
        err = dr // 2
        for _ in range(dr + 1):  # Include endpoint
            if grid[r, c]:
                return False
            if r == r1 and c == c1:
                break
            r += sr
            err -= dc
            if err < 0:
                c += sc
                err += dr
        return True
    else:
        err = dc // 2
        for _ in range(dc + 1):  # Include endpoint
            if grid[r, c]:
                return False
            if r == r1 and c == c1:
                break
            c += sc
            err -= dr
            if err < 0:
                r += sr
                err += dc
        return True

class ThetaStarPlanner:
    def __init__(self, connectivity: int = 8):
        # Theta* is most meaningful with 8-connected grids
        assert connectivity in (4, 8)
        self.conn = connectivity
        if connectivity == 8:
            self.deltas = np.array([
                (-1, -1), (-1, 0), (-1, +1),
                ( 0, -1),          ( 0, +1),
                (+1, -1), (+1, 0), (+1, +1)
            ], dtype=np.int8)
            self.step_costs = np.array([math.sqrt(2), 1, math.sqrt(2),
                                        1,               1,
                                        math.sqrt(2), 1, math.sqrt(2)], dtype=np.float32)
        else:
            self.deltas = np.array([(-1,0), (1,0), (0,-1), (0,1)], dtype=np.int8)
            self.step_costs = np.ones(4, dtype=np.float32)

    def _heuristic(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        dr = abs(a[0] - b[0]); dc = abs(a[1] - b[1])
        if self.conn == 4:
            return float(dr + dc)
        dmin, dmax = (dr, dc) if dr < dc else (dc, dr)
        return float((math.sqrt(2) - 1) * dmin + dmax)

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

        g = np.full((H, W), np.inf, dtype=np.float32)
        f = np.full((H, W), np.inf, dtype=np.float32)
        closed = np.zeros((H, W), dtype=bool)
        par_r = np.full((H, W), -1, dtype=np.int32)
        par_c = np.full((H, W), -1, dtype=np.int32)

        g[sr, sc] = 0.0
        par_r[sr, sc] = sr
        par_c[sr, sc] = sc
        f[sr, sc] = self._heuristic(start, goal)

        pq: List[Tuple[float, int, int]] = []
        heapq.heappush(pq, (f[sr, sc], sr, sc))

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

                # Standard A* tentative cost via current node
                tentative_g = g[r, c] + self.step_costs[k]

                # Theta* shortcut: if LoS between parent(r,c) and (nr,nc), try that parent
                pr, pc = int(par_r[r, c]), int(par_c[r, c])
                # Check if current node has a valid parent (not start node)
                if (pr, pc) != (r, c) and _los_free(grid, (pr, pc), (nr, nc)):
                    # Recompute cost as parent -> neighbor
                    # Distance between (pr,pc) and (nr,nc) in Euclidean metric
                    tentative_g2 = g[pr, pc] + math.hypot(nr - pr, nc - pc)
                    if tentative_g2 < tentative_g:
                        tentative_g = tentative_g2
                        new_parent = (pr, pc)
                    else:
                        new_parent = (r, c)
                else:
                    new_parent = (r, c)

                if tentative_g < g[nr, nc]:
                    g[nr, nc] = tentative_g
                    par_r[nr, nc] = new_parent[0]
                    par_c[nr, nc] = new_parent[1]
                    f[nr, nc] = tentative_g + self._heuristic((nr, nc), goal)
                    heapq.heappush(pq, (f[nr, nc], nr, nc))

        return {'success': False, 'path': None}
