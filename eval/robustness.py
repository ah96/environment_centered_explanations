#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness.py
-------------
Environment perturbations for robustness testing.

Perturbations
-------------
- jitter: randomly translate obstacles by small offsets (keeps shape)
- dilate / erode: morphology on occupancy (may merge/split objects)
- distractors: add small obstacles far from start-goal geodesic line

All functions return a NEW GridEnvironment (same dataclass), rebuilt so that
obj_map/obstacles are consistent after changes.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from explanations.envs.generator import GridEnvironment, rebuild_objects_from_grid
except Exception:
    try:
        from ..envs.generator import GridEnvironment, rebuild_objects_from_grid  # type: ignore
    except Exception:
        from envs.generator import GridEnvironment, rebuild_objects_from_grid  # type: ignore

try:
    from explanations.envs.affordances import move_obstacle
except Exception:
    try:
        from ..envs.affordances import move_obstacle  # type: ignore
    except Exception:
        from envs.affordances import move_obstacle  # type: ignore

# Morphology
try:
    from scipy.ndimage import binary_dilation, binary_erosion
except Exception as e:
    raise ImportError("scipy.ndimage is required for robustness morphology ops") from e


# ------------------------ helpers ------------------------ #

def _bresenham(r0: int, c0: int, r1: int, c1: int):
    pts = []
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        pts.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return pts


def _new_env_from_grid(env: GridEnvironment, new_grid: np.ndarray) -> GridEnvironment:
    new_env = GridEnvironment(
        grid=new_grid.copy(),
        obj_map=env.obj_map.copy(),   # will be replaced in rebuild
        obstacles=list(env.obstacles),
        start=env.start,
        goal=env.goal,
        settings=dict(env.settings),
        rng=env.rng
    )
    new_env = rebuild_objects_from_grid(new_env)
    # Ensure start/goal are free
    new_env.grid[new_env.start] = False
    new_env.grid[new_env.goal] = False
    return new_env


# ---------------------- perturbations -------------------- #

def perturb_jitter(env: GridEnvironment,
                   max_translation: int = 1,
                   n_candidates: int = 10,
                   moat: int = 1,
                   seed: Optional[int] = None) -> GridEnvironment:
    """
    Attempt to translate each obstacle by <= max_translation in r/c (small jitter).
    Keeps shapes intact and avoids overlaps using the affordance's constraints.
    """
    rng = np.random.default_rng(seed)
    # Work on a copy through successive moves
    cur = GridEnvironment(
        grid=env.grid.copy(),
        obj_map=env.obj_map.copy(),
        obstacles=list(env.obstacles),
        start=env.start,
        goal=env.goal,
        settings=dict(env.settings),
        rng=env.rng
    )
    m = len(cur.obstacles)
    # Visit obstacles in random order for variety
    order = rng.permutation(m) + 1
    for oid in order:
        cur, _ = move_obstacle(cur, int(oid),
                               max_translation=max_translation,
                               n_candidates=n_candidates,
                               moat=moat,
                               rng=rng,
                               inplace=True,
                               relabel_after=False)
    # rebuild to sanitize ids/coords
    cur = rebuild_objects_from_grid(cur)
    cur.grid[cur.start] = False
    cur.grid[cur.goal] = False
    return cur


def perturb_dilate(env: GridEnvironment, iterations: int = 1) -> GridEnvironment:
    new_grid = binary_dilation(env.grid, iterations=int(iterations))
    return _new_env_from_grid(env, new_grid)


def perturb_erode(env: GridEnvironment, iterations: int = 1) -> GridEnvironment:
    new_grid = binary_erosion(env.grid, iterations=int(iterations))
    # Ensure start/goal free even if erosion clears area around
    new_grid[env.start] = False
    new_grid[env.goal] = False
    return _new_env_from_grid(env, new_grid)


def perturb_distractors(env: GridEnvironment,
                        n: int = 3,
                        min_l1_dist_to_line: int = 3,
                        seed: Optional[int] = None) -> GridEnvironment:
    """
    Add n small 1x1 'pebble' obstacles placed far from the straight start-goal line.
    """
    rng = np.random.default_rng(seed)
    H, W = env.grid.shape
    line = set(_bresenham(env.start[0], env.start[1], env.goal[0], env.goal[1]))

    new_grid = env.grid.copy()
    added = 0
    trials = 0
    max_trials = 2000
    while added < n and trials < max_trials:
        trials += 1
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, W))
        if new_grid[r, c]:
            continue  # occupied
        if (r, c) in line:
            continue
        # Minimum L1 distance to line
        ok = True
        for lr, lc in line:
            if abs(r - lr) + abs(c - lc) < min_l1_dist_to_line:
                ok = False
                break
        if not ok:
            continue
        new_grid[r, c] = True
        added += 1

    return _new_env_from_grid(env, new_grid)


# ---------------------- suite builder -------------------- #

def robustness_suite(env: GridEnvironment,
                     *,
                     jitter: bool = True,
                     dilate: bool = True,
                     erode: bool = True,
                     distractors: bool = True,
                     seed: Optional[int] = None) -> List[Tuple[str, GridEnvironment]]:
    """
    Build a standard set of perturbed environments for robustness evaluation.
    """
    out: List[Tuple[str, GridEnvironment]] = []
    rng = np.random.default_rng(seed)
    if jitter:
        out.append(("jitter", perturb_jitter(env, max_translation=1, n_candidates=15, moat=1, seed=int(rng.integers(1e9)))))
    if dilate:
        out.append(("dilate", perturb_dilate(env, iterations=1)))
    if erode:
        out.append(("erode", perturb_erode(env, iterations=1)))
    if distractors:
        out.append(("distractors", perturb_distractors(env, n=3, min_l1_dist_to_line=3, seed=int(rng.integers(1e9)))))
    return out
