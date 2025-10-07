#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Permutation-based Shapley value approximation for obstacle-level attribution.

We approximate SHAP by sampling random permutations of features and computing
marginal contributions: phi_i = E_pi[ f(S ∪ {i}) - f(S) ], where S are the
features preceding i in permutation pi. Here, f returns {0,1} success labels.

Implementation notes:
- Subset S is interpreted as the set of *present* obstacles.
- We start from an empty set (no obstacles) and add along the permutation.
- To keep things fast, we build grids incrementally within each permutation.
- We return "harmfulness" = -phi (higher means more responsible for failure).

Returns:
    {
      'ranking': [(obs_id, score), ...],  # descending by harmfulness
      'phi': np.ndarray[m],               # raw Shapley values (positive helps success)
      'calls': int,
      'time_sec': float
    }
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import time
from explainers.baselines import geodesic_line_ranking

def _planner_to_int_success(res) -> int:
    if isinstance(res, dict):
        return int(bool(res.get('success', False)))
    return int(bool(res))


def _empty_grid_like(grid: np.ndarray) -> np.ndarray:
    g = np.zeros_like(grid, dtype=bool)
    return g


def _add_obstacle_inplace(grid: np.ndarray, ob) -> None:
    """Stamp obstacle pixels to True in-place."""
    if ob.coords.size == 0:
        return
    rr, cc = ob.coords[:, 0], ob.coords[:, 1]
    grid[rr, cc] = True



class ShapExplainer:
    """
    Permutation KernelSHAP over binary obstacle features.
    Optional focus_top_m runs SHAP only on top-M obstacles from a cheap geodesic heuristic.
    """
    def __init__(self, permutations: int = 100, random_state: int = None,
                 focus_top_m: int = None):
        self.permutations = int(permutations)
        self.rng = np.random.default_rng(random_state)
        self.focus_top_m = int(focus_top_m) if focus_top_m else None

    def explain(self, env, planner):
        import time
        t0 = time.perf_counter()
        n = len(env.obstacles)
        all_ids = np.array([i for i,o in enumerate(env.obstacles, start=1) if getattr(o, "coords", None) is not None and o.coords.size > 0], dtype=int)

        # ---- Focus subset (optional)
        if self.focus_top_m and self.focus_top_m < n:
            geo = geodesic_line_ranking(env)["ranking"]
            subset_ids = np.array([oid for oid, _ in geo[: self.focus_top_m]], dtype=int)
        else:
            subset_ids = all_ids

        # Precompute masks
        # local index (0..m-1) → global obstacle id
        m = len(subset_ids)
        id_from_local = subset_ids
        # for fast toggle
        obj_map = env.obj_map
        base_grid = env.grid.copy()

        def eval_with_present(present_mask_local: np.ndarray) -> bool:
            # start from original grid; remove only the subset obs that are "absent"
            grid = base_grid.copy()
            # turn OFF the subset obstacles not present
            for j, present in enumerate(present_mask_local):
                if not present:
                    oid = int(id_from_local[j])
                    grid[obj_map == oid] = False
            res = planner.plan(grid, env.start, env.goal)
            return bool(res["success"]) if isinstance(res, dict) else bool(res)

        # KernelSHAP via permutations: contributions for each local feature
        phi_local = np.zeros(m, dtype=float)
        calls = 0

        for _ in range(self.permutations):
            perm = self.rng.permutation(m)
            present = np.zeros(m, dtype=bool)
            y_prev = eval_with_present(present); calls += 1
            for j in perm:
                present[j] = True
                y_cur = eval_with_present(present); calls += 1
                # contribution is change in success prob (bool→int)
                phi_local[j] += (1 if y_cur else 0) - (1 if y_prev else 0)
                y_prev = y_cur

        phi_local /= max(1, self.permutations)

        # Convert to harmfulness scores (bigger ⇒ more harmful)
        harm_local = -phi_local  # if adding obstacle decreases success, harmfulness positive

        # Build full ranking: subset first by score, others appended with very low score
        pairs = [(int(id_from_local[j]), float(harm_local[j])) for j in range(m)]
        # assign a very small score to unconsidered obstacles (deterministic tie-breaker by id)
        if m < n:
            floor = (min(harm_local) if m > 0 else 0.0) - 1e6
            for oid in all_ids:
                if oid not in id_from_local:
                    pairs.append((int(oid), float(floor) - 1e-3 * int(oid)))

        # sort descending by score
        pairs.sort(key=lambda x: x[1], reverse=True)
        return {
            "ranking": pairs,
            "calls": calls,
            "time_sec": time.perf_counter() - t0,
            "considered": int(m),
            "n_total": int(n),
            "focus_top_m": self.focus_top_m or 0,
        }

