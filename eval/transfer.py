#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer.py
-----------
Cross-planner transfer utilities.

Goals
-----
1) Transferability: Does a ranking computed under planner A also work under planner B?
   -> cross_planner_success_at_k()

2) Similarity: How similar are the top-k obstacle sets across planners?
   -> cross_planner_overlap()
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Protocol

# Flexible imports so this works both as a package and as a flat folder
try:
    from envs.generator import GridEnvironment
except Exception:  # pragma: no cover
    try:
        from ..envs.generator import GridEnvironment  # type: ignore
    except Exception:  # pragma: no cover
        from envs.generator import GridEnvironment  # type: ignore

# We reuse helpers from metrics to avoid code duplication
try:
    from eval.metrics import _planner_to_bool, _grid_without, topk_set, jaccard
except Exception:  # pragma: no cover
    try:
        from .metrics import _planner_to_bool, _grid_without, topk_set, jaccard  # type: ignore
    except Exception:
        from metrics import _planner_to_bool, _grid_without, topk_set, jaccard  # type: ignore


class Planner(Protocol):
    """Protocol for planner objects with a plan method."""
    def plan(self, grid, start, goal): ...


def cross_planner_success_at_k(env: "GridEnvironment",
                               ranking_from_A: List[Tuple[int, float]],
                               planner_B: Planner,
                               ks: Iterable[int]) -> Dict[int, int]:
    """
    Apply A's ranking to B: for each k in ks, remove top-k obstacles (by A's ranking)
    and check whether planner B succeeds. Return {k: 0/1}.
    """
    if not ranking_from_A:
        return {int(k): 0 for k in ks}
    
    # Sort by score (descending), extract IDs
    ids_sorted = [i for i, _ in sorted(ranking_from_A, key=lambda x: x[1], reverse=True)]
    out: Dict[int, int] = {}
    for k in ks:
        k_int = int(k)
        subset = ids_sorted[:k_int]
        gb = _grid_without(env, subset)
        succ = _planner_to_bool(planner_B.plan(gb, env.start, env.goal))
        out[k_int] = int(succ)
    return out


def cross_planner_overlap(ranking_A: List[Tuple[int, float]],
                          ranking_B: List[Tuple[int, float]],
                          k: int) -> float:
    """
    Jaccard overlap of top-k obstacle sets from two planners' explanations.
    """
    if not ranking_A or not ranking_B:
        return 0.0
    
    SA = topk_set(ranking_A, k)
    SB = topk_set(ranking_B, k)
    return jaccard(SA, SB)


__all__ = [
    "cross_planner_success_at_k",
    "cross_planner_overlap",
]