#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-First Search planner (not optimal, but useful as a baseline).
- 4- or 8-connected grids.
- Returns the first path found (often long and twisty).
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np


class DFSPlanner:
    def __init__(self, connectivity: int = 8):
        assert connectivity in (4, 8)
        self.conn = connectivity
        if connectivity == 8:
            self.deltas = [
                (-1, -1), (-1, 0), (-1, +1),
                ( 0, -1),          ( 0, +1),
                (+1, -1), (+1, 0), (+1, +1)
            ]
        else:
            self.deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    @staticmethod
    def _reconstruct(par_r: np.ndarray, par_c: np.ndarray,
                     start: Tuple[int, int], goal: Tuple[int, int]):
        """Reconstruct path from parent arrays."""
        if par_r[goal] == -1 and goal != start:
            return None
        path = []
        r, c = goal
        while (r, c) != start:
            path.append((int(r), int(c)))
            pr, pc = int(par_r[r, c]), int(par_c[r, c])
            if pr == -1 and pc == -1:
                return None
            r, c = pr, pc
        path.append(start)
        path.reverse()
        return path

    def plan(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict:
        """
        Find a path from start to goal using depth-first search.
        
        Args:
            grid: 2D numpy array where 0=free, non-zero=obstacle
            start: (row, col) starting position
            goal: (row, col) goal position
            
        Returns:
            Dictionary with 'success' (bool) and 'path' (list or None)
        """
        H, W = grid.shape
        sr, sc = start
        gr, gc = goal
        
        # Validate start and goal
        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            return {'success': False, 'path': None}
        
        if grid[sr, sc] or grid[gr, gc]:
            return {'success': False, 'path': None}
        
        if start == goal:
            return {'success': True, 'path': [start]}

        visited = np.zeros((H, W), dtype=bool)
        par_r = np.full((H, W), -1, dtype=np.int32)
        par_c = np.full((H, W), -1, dtype=np.int32)

        stack = [(sr, sc)]
        visited[sr, sc] = True

        while stack:
            r, c = stack.pop()
            if (r, c) == (gr, gc):
                path = self._reconstruct(par_r, par_c, start, goal)
                return {'success': True, 'path': path}
            
            # Explore neighbors
            for dr, dc in self.deltas:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if visited[nr, nc] or grid[nr, nc]:
                    continue
                visited[nr, nc] = True
                par_r[nr, nc] = r
                par_c[nr, nc] = c
                stack.append((nr, nc))

        return {'success': False, 'path': None}