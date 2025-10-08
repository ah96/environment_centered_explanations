#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py
----------
Metrics for attribution, intervention, robustness, and efficiency.

Assumptions
-----------
- Environment: GridEnvironment from generator.py
- Planner API: planner.plan(grid, start, goal) -> {'success': bool, 'path': ...} or bool
- Explainer outputs:
    * Ranking-based (LIME/SHAP/baselines): {'ranking': [(obs_id, score), ...], 'calls', 'time_sec', ...}
    * COSE: {'cose_set': set[int], 'calls', 'time_sec', ...}

What's inside
-------------
- success_at_k() / auc_success_at_k() for faithfulness
- topk_set() helper + jaccard() for set overlap
- kendall_tau() for ranking stability
- evaluate_ranking_success_curve(): convenience wrapper for one env
- runtime helpers
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional, Any, Union
import numpy as np
import math
import time

# Flexible imports so this works whether used as a package or simple folder
try:
    from envs.generator import GridEnvironment
except Exception:
    try:
        from ..envs.generator import GridEnvironment  # type: ignore
    except Exception:
        from envs.generator import GridEnvironment  # type: ignore


def _planner_to_bool(res: Union[Dict, bool]) -> bool:
    if isinstance(res, dict):
        return bool(res.get('success', False))
    return bool(res)


def _grid_without(env: GridEnvironment, remove_ids: Iterable[int]) -> np.ndarray:
    g = env.grid.copy()
    grid_height, grid_width = g.shape
    
    for oid in set(int(i) for i in remove_ids):
        if 1 <= oid <= len(env.obstacles):
            ob = env.obstacles[oid - 1]
            if ob.coords.size > 0:
                rr, cc = ob.coords[:, 0], ob.coords[:, 1]
                # Validate coordinates are within bounds
                valid_mask = (rr >= 0) & (rr < grid_height) & (cc >= 0) & (cc < grid_width)
                rr, cc = rr[valid_mask], cc[valid_mask]
                if len(rr) > 0:
                    g[rr, cc] = False
    # keep start/goal free
    if 0 <= env.start[0] < grid_height and 0 <= env.start[1] < grid_width:
        g[env.start] = False
    if 0 <= env.goal[0] < grid_height and 0 <= env.goal[1] < grid_width:
        g[env.goal] = False
    return g


# -------------------- Faithfulness: Success@k and AUC-S@K -------------------- #

def success_at_k(env: GridEnvironment,
                 planner: Any,
                 ranking: List[Tuple[int, float]],
                 ks: Iterable[int]) -> Dict[int, int]:
    """
    For each k, remove top-k obstacles (by ranking) and check if planner succeeds.
    Returns a dict {k: 0/1}. (Use mean across envs for rates.)
    """
    out: Dict[int, int] = {}
    ids_sorted = [int(i) for i, _ in sorted(ranking, key=lambda x: (-x[1], x[0]))]
    for k in ks:
        subset = ids_sorted[:int(k)]
        gk = _grid_without(env, subset)
        succ = _planner_to_bool(planner.plan(gk, env.start, env.goal))
        out[int(k)] = int(succ)
    return out


def auc_success_at_k(succ_curve: Dict[int, int]) -> float:
    """
    AUC-S@K: normalized area under the Success@k curve (discrete mean).
    Expects succ_curve with integer keys k=1..K. Returns in [0,1].
    """
    if not succ_curve:
        return 0.0
    ks = sorted(succ_curve.keys())
    vals = [float(succ_curve[k]) for k in ks]
    return float(np.mean(vals))


# ------------------------- Set overlap & rank correlation -------------------- #

def topk_set(ranking: List[Tuple[int, float]], k: int) -> set:
    ids_sorted = [int(i) for i, _ in sorted(ranking, key=lambda x: (-x[1], x[0]))]
    return set(ids_sorted[:int(k)])


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    union = len(A | B)
    if union == 0:
        return 0.0
    return len(A & B) / union


def kendall_tau(ranking_a: List[Tuple[int, float]],
                ranking_b: List[Tuple[int, float]]) -> float:
    """
    Simple Kendall's tau (not tau-b). Ties are broken deterministically by id.
    Returns in [-1, 1]. Works for m up to a few hundred.
    """
    # Map id -> rank (1 best). Break ties by (score desc, id asc)
    order_a = sorted(ranking_a, key=lambda x: (-x[1], x[0]))
    order_b = sorted(ranking_b, key=lambda x: (-x[1], x[0]))
    id_to_rank_a = {int(oid): idx + 1 for idx, (oid, _) in enumerate(order_a)}
    id_to_rank_b = {int(oid): idx + 1 for idx, (oid, _) in enumerate(order_b)}
    # Use intersection of ids present in both rankings
    ids = sorted(set(id_to_rank_a) & set(id_to_rank_b))
    n = len(ids)
    if n <= 1:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        id_i = ids[i]
        for j in range(i + 1, n):
            id_j = ids[j]
            ai, aj = id_to_rank_a[id_i], id_to_rank_a[id_j]
            bi, bj = id_to_rank_b[id_i], id_to_rank_b[id_j]
            s1 = 1 if ai < aj else -1 if ai > aj else 0
            s2 = 1 if bi < bj else -1 if bi > bj else 0
            if s1 * s2 > 0:
                concordant += 1
            elif s1 * s2 < 0:
                discordant += 1
            # if both zero (ties both sides), ignore
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return float(concordant - discordant) / float(denom)


# ------------------------------- Convenience -------------------------------- #

def evaluate_ranking_success_curve(env: GridEnvironment,
                                   planner: Any,
                                   ranking: List[Tuple[int, float]],
                                   k_max: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience wrapper: compute Success@k for k=1..K and AUC-S@K.
    """
    m = len(env.obstacles)
    if m == 0:
        return {'success_at_k': {}, 'auc_s_at_k': 0.0}
    K = m if k_max is None else int(min(k_max, m))
    ks = list(range(1, K + 1))
    curve = success_at_k(env, planner, ranking, ks)
    aucv = auc_success_at_k(curve)
    return {'success_at_k': curve, 'auc_s_at_k': float(aucv)}


# ------------------------------ Runtime helpers ----------------------------- #

def sum_calls(*records: Dict) -> int:
    """Sum the 'calls' fields from explainer outputs (ignore missing)."""
    total = 0
    for r in records:
        total += int(r.get('calls', 0))
    return total


def sum_time_sec(*records: Dict) -> float:
    """Sum 'time_sec' fields from explainer outputs (ignore missing)."""
    total = 0.0
    for r in records:
        total += float(r.get('time_sec', 0.0))
    return total