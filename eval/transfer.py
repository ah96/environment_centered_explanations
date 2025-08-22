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
from typing import Dict, List, Tuple, Iterable

# Flexible imports so this works both as a package and as a flat folder
try:
    from explanations.envs.generator import GridEnvironment
except Exception:  # pragma: no cover
    try:
        from ..envs.generator import GridEnvironment  # type: ignore
    except Exception:  # pragma: no cover
        from envs.generator import GridEnvironment  # type: ignore

# We reuse helpers from metrics to avoid code duplication
try:
    from explanations.eval.metrics import _planner_to_bool, _grid_without, topk_set, jaccard
except Exception:  # pragma: no cover
    try:
        from .metrics import _planner_to_bool, _grid_without, topk_set, jaccard  # type: ignore
    except Exception:
        from metrics import _planner_to_bool, _grid_without, topk_set, jaccard  # type: ignore


def cross_planner_success_at_k(env: "GridEnvironment",
                               ranking_from_A: List[Tuple[int, float]],
                               planner_B,
                               ks: Iterable[int]) -> Dict[int, int]:
    """
    Apply A's ranking to B: for each k in ks, remove top-k obstacles (by A's ranking)
    and check whether planner B succeeds. Return {k: 0/1}.
    """
    ids_sorted = [int(i) for i, _ in sorted(ranking_from_A, key=lambda x: x[1], reverse=True)]
    out: Dict[int, int] = {}
    for k in ks:
        subset = ids_sorted[:int(k)]
        gb = _grid_without(env, subset)
        succ = _planner_to_bool(planner_B.plan(gb, env.start, env.goal))
        out[int(k)] = int(succ)
    return out


def cross_planner_overlap(ranking_A: List[Tuple[int, float]],
                          ranking_B: List[Tuple[int, float]],
                          k: int) -> float:
    """
    Jaccard overlap of top-k obstacle sets from two planners' explanations.
    """
    SA = topk_set(ranking_A, k)
    SB = topk_set(ranking_B, k)
    return jaccard(SA, SB)


__all__ = [
    "cross_planner_success_at_k",
    "cross_planner_overlap",
]
