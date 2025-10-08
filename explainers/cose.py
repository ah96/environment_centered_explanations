#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COSE: Critical Obstacle Set Explainer (greedy + redundancy pruning).

Goal: compute a *minimal counterfactual* set O_c such that removing O_c turns
a failure into a success. We use a simple two-stage procedure:

1) Greedy forward selection:
   Traverse obstacles in a given order (optionally guided by a ranking from
   LIME/SHAP/baseline). Iteratively add obstacles to the removal set until
   success is achieved.

2) Redundancy pruning:
   Try removing each selected obstacle from the set; if success persists, drop it.
   This yields an *irreducible* set (no proper subset suffices), which is a
   standard minimality notion for counterfactuals.

Returns:
    {
      'cose_set': set[int],    # 1-based obstacle ids
      'order_used': list[int], # attempted order (1-based)
      'calls': int,
      'time_sec': float
    }
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Dict, Set
import numpy as np
import time


def _planner_to_int_success(res) -> int:
    if isinstance(res, dict):
        return int(bool(res.get('success', False)))
    return int(bool(res))

def _grid_without(env, trial):
    """
    Return a copy of env.grid where all obstacles in `trial` are removed.
    `trial` can contain obstacle *IDs* (1-based) or objects with `.coords`.
    """
    G = np.array(env.grid, dtype=bool).copy()
    obj_map = getattr(env, "obj_map", None)
    for ob in trial:
        # Obstacle ID (preferred path)
        if isinstance(ob, (int, np.integer)):
            oid = int(ob)
            if obj_map is not None and oid > 0:
                G[obj_map == oid] = False
            continue
        # Object with coords
        if hasattr(ob, "coords") and getattr(ob, "coords") is not None:
            coords = np.asarray(ob.coords)
            if coords.size > 0:
                G[coords[:, 0], coords[:, 1]] = False
            continue
        # Fallback: ignore unsupported entries
    return G



class COSEExplainer:
    """
    Parameters
    ----------
    guide : Optional[str]
        One of {'lime', 'shap', 'none'}. Only used for metadata; you actually
        pass the ranking list via 'guide_ranking' to explain().
    max_remove : Optional[int]
        Optional safety cap on the number of obstacles to remove during greedy phase.
    """

    def __init__(self, guide: str = 'none', max_remove: Optional[int] = None):
        assert guide in {'none', 'lime', 'shap'}
        self.guide = guide
        self.max_remove = max_remove

    def explain(self,
                env,
                planner,
                *,
                guide_ranking: Optional[List[Tuple[int, float]]] = None) -> Dict:
        """
        Compute a minimal counterfactual removal set.

        guide_ranking: optional list of (obs_id, score). Higher score tried first.
                       If None, uses obstacle ids in increasing order.
        """
        t0 = time.perf_counter()
        calls = 0

        # Quick check: if already success, empty set is minimal.
        result = planner.plan(env.grid.copy(), env.start, env.goal)
        calls += 1
        if _planner_to_int_success(result) == 1:
            return {'cose_set': set(), 'order_used': [], 'calls': calls, 'time_sec': time.perf_counter() - t0}

        # Build trial order
        if guide_ranking is not None and len(guide_ranking) > 0:
            order = [int(i) for (i, _) in sorted(guide_ranking, key=lambda x: x[1], reverse=True)]
        else:
            order = list(range(1, len(env.obstacles) + 1))

        # 1) Greedy forward selection: add obstacles until success
        selected: List[int] = []
        success_found = False
        
        for oid in order:
            if self.max_remove is not None and len(selected) >= self.max_remove:
                break
            
            trial = selected + [oid]
            grid = _grid_without(env, trial)
            result = planner.plan(grid, env.start, env.goal)
            succ = _planner_to_int_success(result)
            calls += 1
            
            selected.append(oid)
            
            if succ == 1:
                success_found = True
                break  # success achieved; proceed to pruning

        # If still not successful after greedy phase, return the selected set
        if not success_found:
            grid = _grid_without(env, selected)
            result = planner.plan(grid, env.start, env.goal)
            calls += 1
            if _planner_to_int_success(result) == 0:
                t1 = time.perf_counter()
                return {'cose_set': set(selected), 'order_used': order, 'calls': calls, 'time_sec': t1 - t0}

        # 2) Redundancy pruning: remove any obstacle that is not necessary.
        pruned = list(selected)
        i = 0
        while i < len(pruned):
            oid = pruned[i]
            test_set = [x for x in pruned if x != oid]
            grid = _grid_without(env, test_set)
            result = planner.plan(grid, env.start, env.goal)
            succ = _planner_to_int_success(result)
            calls += 1
            
            if succ == 1:
                # This obstacle is redundant, remove it
                pruned = test_set
                # Don't increment i, check the same position again
            else:
                # This obstacle is necessary, keep it and move to next
                i += 1

        t1 = time.perf_counter()
        return {
            'cose_set': set(pruned),
            'order_used': order,
            'calls': calls,
            'time_sec': t1 - t0
        }