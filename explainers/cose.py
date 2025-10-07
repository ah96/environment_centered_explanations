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
        if _planner_to_int_success(planner.plan(env.grid.copy(), env.start, env.goal)) == 1:
            return {'cose_set': set(), 'order_used': [], 'calls': 1, 'time_sec': time.perf_counter() - t0}

        # Build trial order
        if guide_ranking is not None and len(guide_ranking) > 0:
            order = [int(i) for (i, _) in sorted(guide_ranking, key=lambda x: x[1], reverse=True)]
        else:
            order = list(range(1, len(env.obstacles) + 1))

        # 1) Greedy forward selection: add until success
        selected: List[int] = []
        for oid in order:
            if self.max_remove is not None and len(selected) >= self.max_remove:
                break
            trial = selected + [oid]
            grid = _grid_without(env, trial)
            succ = _planner_to_int_success(planner.plan(grid, env.start, env.goal))
            calls += 1
            if succ == 1:
                selected = trial
                break  # success achieved; proceed to pruning
            else:
                # Keep the obstacle ONLY if it showed progress? We cannot measure progress
                # without a numeric metric; in practice, we collect obstacles until success.
                selected = trial

        # If still not successful (e.g., order exhausted), keep trying all remaining
        idx = 0
        while _planner_to_int_success(planner.plan(_grid_without(env, selected), env.start, env.goal)) == 0 and idx < len(order):
            calls += 1
            # Add the next not-yet-selected obstacle
            for oid in order:
                if oid not in selected:
                    selected.append(oid)
                    break
            idx += 1

        # If still failure after exhausting obstacles, return full set (degenerate)
        if _planner_to_int_success(planner.plan(_grid_without(env, selected), env.start, env.goal)) == 0:
            calls += 1
            t1 = time.perf_counter()
            return {'cose_set': set(selected), 'order_used': order, 'calls': calls, 'time_sec': t1 - t0}

        calls += 1  # last check above

        # 2) Redundancy pruning: remove any obstacle that is not necessary.
        pruned = list(selected)
        changed = True
        while changed:
            changed = False
            for oid in list(pruned):
                test_set = [x for x in pruned if x != oid]
                grid = _grid_without(env, test_set)
                succ = _planner_to_int_success(planner.plan(grid, env.start, env.goal))
                calls += 1
                if succ == 1:
                    pruned = test_set
                    changed = True

        t1 = time.perf_counter()
        return {
            'cose_set': set(pruned),
            'order_used': order,
            'calls': calls,
            'time_sec': t1 - t0
        }
