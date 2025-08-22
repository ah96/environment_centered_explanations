#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baselines that do not query the planner.
- random_ranking(env, planner=None, random_state=None)
- geodesic_line_ranking(env, planner=None)
Return a dict with: {'ranking': [(obs_id, score), ...], 'calls': 0, 'time_sec': float}
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import time

# Small helper: Bresenham line between two grid cells
def _bresenham(r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    """Return array of (r,c) points on the discrete line from (r0,c0) to (r1,c1)."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sgn_r = 1 if r0 < r1 else -1
    sgn_c = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    pts = [(r, c)]
    while (r, c) != (r1, c1):
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sgn_r
        if e2 < dr:
            err += dr
            c += sgn_c
        pts.append((r, c))
    return np.array(pts, dtype=int)

def random_ranking(env, planner=None, *, random_state: Optional[int] = None) -> Dict:
    """
    Uniform random permutation of obstacle ids (1..n).
    planner is accepted for API compatibility but ignored.
    """
    t0 = time.perf_counter()
    n = len(env.obstacles)
    ids = np.arange(1, n + 1, dtype=int)
    rng = np.random.default_rng(random_state)
    rng.shuffle(ids)
    # give decreasing scores so higher = “more harmful”
    scores = np.linspace(1.0, 0.0, num=n, endpoint=False)
    ranking: List[Tuple[int, float]] = [(int(i), float(s)) for i, s in zip(ids, scores)]
    return {"ranking": ranking, "calls": 0, "time_sec": time.perf_counter() - t0}

def geodesic_line_ranking(env, planner=None) -> Dict:
    """
    Rank obstacles by proximity to the straight start→goal line (Bresenham).
    Heuristic: obstacles whose pixels lie closer to that line get higher scores.
    planner is accepted for API compatibility but ignored.
    """
    t0 = time.perf_counter()
    (r0, c0), (r1, c1) = env.start, env.goal
    line = _bresenham(r0, c0, r1, c1)
    # distance grid: for each cell, min L1 distance to any line pixel
    H, W = env.grid.shape
    dist = np.full((H, W), np.inf, dtype=float)
    for (rr, cc) in line:
        # cheap L1 expansion around (rr,cc)
        # update a box; this is faster than full cdist for small maps
        rmin = 0; rmax = H; cmin = 0; cmax = W
        rs = np.arange(rmin, rmax)[:, None]
        cs = np.arange(cmin, cmax)[None, :]
        # update with |r-rr|+|c-cc|
        cur = np.abs(rs - rr) + np.abs(cs - cc)
        dist = np.minimum(dist, cur)
    # score each obstacle: negative of min distance (closer ⇒ bigger score)
    ranking: List[Tuple[int, float]] = []
    for oid, ob in enumerate(env.obstacles, start=1):
        if ob.coords.size == 0:
            ranking.append((oid, -1e9))  # effectively last
            continue
        rr, cc = ob.coords[:, 0], ob.coords[:, 1]
        dmin = float(np.min(dist[rr, cc]))
        ranking.append((oid, -dmin))
    # sort by score descending already handled by downstream, but keep as-is
    return {"ranking": ranking, "calls": 0, "time_sec": time.perf_counter() - t0}
