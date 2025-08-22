#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ilp_minimality.py
-----------------
Exact (oracle) minimal obstacle removal set for *small* grids via MILP (PuLP).

Idea
----
Let each grid cell be a node in a flow network. A cell is traversable if it is
originally free OR it belongs to an obstacle that we remove. Send 1 unit of flow
from start to goal through traversable cells. Minimize number of removed obstacles.

Variables
---------
x_i ∈ {0,1} : remove obstacle i (1) or keep (0)
t_v ∈ {0,1} : cell v is traversable
f_uv ≥ 0    : flow on directed edge u->v

Constraints
-----------
- If cell v is free: t_v = 1
- If cell v in obstacle i: t_v = x_i
- Edge capacity: f_uv ≤ t_u, f_uv ≤ t_v, and only for neighbor pairs
- Flow conservation with unit demand from start to goal.

Objective
---------
minimize sum_i x_i

Returns
-------
{'set': set[int], 'status': 'Optimal'|'TimeLimit'|..., 'obj': float, 'time_sec': float}
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Set, Optional
import numpy as np
import time

# Flexible import
try:
    from explanations.envs.generator import GridEnvironment
except Exception:
    try:
        from ..envs.generator import GridEnvironment  # type: ignore
    except Exception:
        from envs.generator import GridEnvironment  # type: ignore

# PuLP MILP
try:
    import pulp
except Exception as e:
    raise ImportError("PuLP is required for ilp_minimality. Install with: pip install pulp") from e


def _neighbors(H: int, W: int, connectivity: int = 8):
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(-1, -1), (-1, 0), (-1, +1),
                  (0, -1),           (0, +1),
                  (+1, -1), (+1, 0), (+1, +1)]
    for r in range(H):
        for c in range(W):
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    yield (r, c, nr, nc)


def exact_minimal_set(env: GridEnvironment,
                      *,
                      connectivity: int = 8,
                      time_limit: Optional[int] = 60,
                      solver: Optional[pulp.LpSolver] = None) -> Dict:
    t0 = time.perf_counter()
    H, W = env.grid.shape
    n_obs = len(env.obstacles)

    # Problem
    prob = pulp.LpProblem("MinimalObstacleRemoval", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts('x', (i + 1 for i in range(n_obs)), 0, 1, cat=pulp.LpBinary)  # obstacle removal
    t = pulp.LpVariable.dicts('t', ((r, c) for r in range(H) for c in range(W)), 0, 1, cat=pulp.LpBinary)  # traversable
    f = pulp.LpVariable.dicts('f', ((r, c, nr, nc) for (r, c, nr, nc) in _neighbors(H, W, connectivity)),
                              lowBound=0, upBound=1, cat=pulp.LpContinuous)

    # Objective: minimize number of removed obstacles
    prob += pulp.lpSum([x[i + 1] for i in range(n_obs)])

    # Traversability constraints
    # Free cells -> t=1; occupied cell in obstacle i -> t = x_i
    for r in range(H):
        for c in range(W):
            if not env.grid[r, c]:
                prob += (t[(r, c)] == 1), f"free_cell_{r}_{c}"
            else:
                oid = int(env.obj_map[r, c])
                prob += (t[(r, c)] == x[oid]), f"occ_cell_{r}_{c}_oid_{oid}"

    # Edge capacity: f_uv <= t_u and f_uv <= t_v
    for (r, c, nr, nc) in _neighbors(H, W, connectivity):
        prob += (f[(r, c, nr, nc)] <= t[(r, c)]), f"cap_from_{r}_{c}_to_{nr}_{nc}"
        prob += (f[(r, c, nr, nc)] <= t[(nr, nc)]), f"cap_to_{r}_{c}_to_{nr}_{nc}"

    # Flow conservation
    sr, sc = env.start
    gr, gc = env.goal
    for r in range(H):
        for c in range(W):
            inflow = pulp.lpSum([f[(nr, nc, r, c)] for (nr, nc, rr, cc) in _neighbors(H, W, connectivity) if rr == r and cc == c])
            outflow = pulp.lpSum([f[(r, c, nr, nc)] for (rr, cc, nr, nc) in _neighbors(H, W, connectivity) if rr == r and cc == c])
            if (r, c) == (sr, sc):
                prob += (outflow - inflow == 1), f"flow_source_{r}_{c}"
            elif (r, c) == (gr, gc):
                prob += (outflow - inflow == -1), f"flow_sink_{r}_{c}"
            else:
                prob += (outflow - inflow == 0), f"flow_cons_{r}_{c}"

    # Solver
    solver = solver or pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status = prob.solve(solver)

    # Extract solution
    sel: Set[int] = set()
    for i in range(n_obs):
        xi = pulp.value(x[i + 1])
        if xi is not None and xi > 0.5:
            sel.add(i + 1)

    status_str = pulp.LpStatus.get(status, "Unknown")
    obj_val = pulp.value(prob.objective)

    t1 = time.perf_counter()
    return {'set': sel, 'status': status_str, 'obj': float(obj_val) if obj_val is not None else float('nan'),
            'time_sec': t1 - t0}
