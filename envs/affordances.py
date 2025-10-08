#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
affordances.py
--------------
Environment perturbation affordances for object-level obstacles:
- remove: eliminate given obstacle(s)
- move  : relocate one obstacle to a new, non-overlapping position

These operations are *planner-agnostic*: they only modify the occupancy grid
and the object map. They preserve object semantics (ids) when done in-place.

Design:
- In-place and functional styles: all functions accept `inplace` (default False).
- Moat enforcement: by default, maintain a 1-cell separation between *different*
  objects after the move (configurable).
- Efficient stamping: vectorized boolean ops; minimal recomputation of obj_map.
- Robustness: after structural changes, we can optionally re-label connected
  components to ensure each obstacle remains a single connected set (rarely
  needed if moves avoid touching).

Dependencies:
    numpy
    scipy.ndimage  (binary_dilation)
    generator.py   (GridEnvironment, Obstacle)

Usage (quick smoke test):
    python3 affordances.py

Author: RAL Paper Toolkit
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import binary_dilation
except Exception as e:
    raise ImportError(
        "scipy.ndimage is required. Install with: pip install scipy"
    ) from e

# Local imports
from envs.generator import GridEnvironment, Obstacle, rebuild_objects_from_grid
# Optional helper only if your generator defines it; safe fallback otherwise.
try:
    from envs.generator import _dilate_bool  # type: ignore
except Exception:
    _dilate_bool = None  # not required unless you call it


# ------------------------------- Remove affordance -------------------------- #

def remove_obstacles(env: GridEnvironment,
                     obstacle_ids: Iterable[int],
                     *,
                     inplace: bool = False,
                     relabel_after: bool = False) -> GridEnvironment:
    """
    Remove (delete) the specified obstacles from the environment.

    Parameters
    ----------
    env : GridEnvironment
    obstacle_ids : Iterable[int]
        Obstacle ids (1-based) to remove.
    inplace : bool
        If True, modify env in place and return it. Otherwise, return a copy.
    relabel_after : bool
        If True, recompute connected components to refresh obj_map/obstacles.
        Typically not necessary for removals (structure only gets sparser),
        but can be enabled to keep ids tight (1..K) after many ops.

    Returns
    -------
    GridEnvironment
        Modified environment (same object instance if inplace=True).
    """
    if inplace:
        new_env = env
    else:
        new_env = GridEnvironment(
            grid=env.grid.copy(),
            obj_map=env.obj_map.copy(),
            obstacles=list(env.obstacles),  # shallow copy ok; will update selectively
            start=env.start, goal=env.goal,
            settings=dict(env.settings),
            rng=env.rng
        )

    H, W = new_env.shape
    to_remove = set(int(i) for i in obstacle_ids)

    if not to_remove:
        return new_env

    # Clear grid + obj_map pixels for each selected obstacle
    for i in sorted(to_remove):
        if i < 1 or i > len(new_env.obstacles):
            continue  # invalid id, ignore
        ob = new_env.obstacles[i - 1]
        rr, cc = ob.coords[:, 0], ob.coords[:, 1]
        new_env.grid[rr, cc] = False
        new_env.obj_map[rr, cc] = 0

    # Optionally rebuild full structure (compact ids)
    if relabel_after:
        new_env = rebuild_objects_from_grid(new_env)
    else:
        # Mark removed obstacles as empty to prevent accidental reuse
        # (ids remain but coords are now empty).
        for i in sorted(to_remove):
            if 1 <= i <= len(new_env.obstacles):
                new_env.obstacles[i - 1] = Obstacle(id=i, coords=np.zeros((0, 2), dtype=int),
                                                    bbox=(0, 0, 0, 0))

    # Ensure start/goal are free
    new_env.grid[new_env.start] = False
    new_env.grid[new_env.goal] = False

    return new_env


# ------------------------------- Move affordance ---------------------------- #

def _stamp_object_at(env: GridEnvironment,
                     ob: Obstacle,
                     top_left: Tuple[int, int],
                     *,
                     moat: int,
                     allow_touch: bool = False) -> bool:
    """
    Try to stamp 'ob' shape at the new top-left, enforcing moat separation from
    *other* objects. Returns True if successful. (No copies; the caller handles
    grid updates.)
    """
    H, W = env.shape
    # Build a local mask for the object's bounding box shape
    r0, r1, c0, c1 = ob.bbox
    h, w = (r1 - r0), (c1 - c0)
    if h <= 0 or w <= 0:
        return False
    mask = np.zeros((h, w), dtype=bool)
    # Normalize coords within bbox
    rr = ob.coords[:, 0] - r0
    cc = ob.coords[:, 1] - c0
    if rr.size == 0:
        return False
    mask[rr, cc] = True

    nr0, nc0 = top_left
    nr1, nc1 = nr0 + h, nc0 + w
    if nr0 < 0 or nc0 < 0 or nr1 > H or nc1 > W:
        return False

    # Extract target slice excluding the object's own current pixels (we clear first in caller)
    target = env.grid[nr0:nr1, nc0:nc1]
    if (target & mask).any():
        return False

    if not allow_touch and moat > 0:
        # Dilate the object's local mask by `moat` cells
        if _dilate_bool is not None:
            dil = _dilate_bool(mask, iters=moat)
        else:
            # Use scipy's binary_dilation with proper structure
            structure = np.ones((3, 3), dtype=bool)
            dil = binary_dilation(mask, structure=structure, iterations=moat)

        # Padded ROI on the grid (clamped to image bounds)
        rpad0 = max(0, nr0 - moat)
        cpad0 = max(0, nc0 - moat)
        rpad1 = min(H, nr1 + moat)
        cpad1 = min(W, nc1 + moat)
        grid_roi = env.grid[rpad0:rpad1, cpad0:cpad1]

        # Canvas that matches grid_roi shape
        canvas_h = rpad1 - rpad0
        canvas_w = cpad1 - cpad0
        ring_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)

        # Where to insert the dilated mask inside the canvas
        ins_r0 = nr0 - rpad0
        ins_c0 = nc0 - cpad0
        ins_r1 = min(canvas_h, ins_r0 + dil.shape[0])
        ins_c1 = min(canvas_w, ins_c0 + dil.shape[1])

        # Corresponding slice from the dilated mask (handle negative offsets)
        src_r0 = max(0, -ins_r0)
        src_c0 = max(0, -ins_c0)
        # Adjust insertion bounds if negative
        ins_r0 = max(0, ins_r0)
        ins_c0 = max(0, ins_c0)
        
        src_r1 = src_r0 + (ins_r1 - ins_r0)
        src_c1 = src_c0 + (ins_c1 - ins_c0)

        # Ensure slices are valid
        if src_r1 > dil.shape[0]:
            src_r1 = dil.shape[0]
            ins_r1 = ins_r0 + (src_r1 - src_r0)
        if src_c1 > dil.shape[1]:
            src_c1 = dil.shape[1]
            ins_c1 = ins_c0 + (src_c1 - src_c0)

        # Paste the dilated mask into the ring_canvas
        if ins_r1 > ins_r0 and ins_c1 > ins_c0 and src_r1 > src_r0 and src_c1 > src_c0:
            ring_canvas[ins_r0:ins_r1, ins_c0:ins_c1] = dil[src_r0:src_r1, src_c0:src_c1]

        # If the dilated ring touches any existing obstacle, reject placement
        if (grid_roi & ring_canvas).any():
            return False

    # OK to stamp here
    env.grid[nr0:nr1, nc0:nc1] |= mask
    # Update obj_map and obstacle coords/bbox
    # First, compute absolute coords of the new placement:
    abs_rr = rr + nr0
    abs_cc = cc + nc0
    new_coords = np.column_stack((abs_rr, abs_cc))

    # Write id into obj_map
    env.obj_map[abs_rr, abs_cc] = ob.id

    # Update the obstacle object
    new_bbox = (nr0, nr1, nc0, nc1)
    env.obstacles[ob.id - 1] = Obstacle(id=ob.id, coords=new_coords, bbox=new_bbox)
    return True


def move_obstacle(env: GridEnvironment,
                  obstacle_id: int,
                  *,
                  max_translation: int = 5,
                  n_candidates: int = 40,
                  moat: int = 1,
                  rng: Optional[np.random.Generator] = None,
                  inplace: bool = False,
                  relabel_after: bool = False) -> Tuple[GridEnvironment, bool]:
    """
    Attempt to move an obstacle by sampling candidate translations in a radius.
    Keeps shape intact and enforces non-overlap and moat wrt. *other* objects.

    Parameters
    ----------
    env : GridEnvironment
    obstacle_id : int
        Id of the object to move (1-based).
    max_translation : int
        Maximum |dr|,|dc| for random translations sampled uniformly in
        [-max_translation, max_translation].
    n_candidates : int
        How many candidate placements to try.
    moat : int
        Minimum free-cell moat to maintain vs. other objects after the move.
    rng : np.random.Generator
        RNG for reproducibility. If None, uses env.rng.
    inplace : bool
        If True, operate in-place, else work on a deep copy of grid/obj_map/ob.
    relabel_after : bool
        If True, recompute connected components from scratch (slower). Usually
        unnecessary if moves enforce non-touching.

    Returns
    -------
    (GridEnvironment, bool)
        The environment (same instance if inplace=True), and a flag indicating
        whether a valid move was performed.
    """
    rng = rng or env.rng

    if inplace:
        new_env = env
    else:
        # Deep copies for safe functional style
        new_env = GridEnvironment(
            grid=env.grid.copy(),
            obj_map=env.obj_map.copy(),
            obstacles=list(env.obstacles),
            start=env.start, goal=env.goal,
            settings=dict(env.settings),
            rng=env.rng
        )

    if obstacle_id < 1 or obstacle_id > len(new_env.obstacles):
        return new_env, False

    ob = new_env.obstacles[obstacle_id - 1]
    if ob.coords.size == 0:
        return new_env, False

    # Clear current object footprint from grid/obj_map
    rr, cc = ob.coords[:, 0], ob.coords[:, 1]
    new_env.grid[rr, cc] = False
    new_env.obj_map[rr, cc] = 0

    # Candidate sampling around the original bbox top-left
    r0, r1, c0, c1 = ob.bbox
    base_tl = (r0, c0)
    H, W = new_env.shape

    # Precompute a small pool of candidate offsets (uniform in square radius)
    candidates = []
    for _ in range(n_candidates):
        dr = int(rng.integers(-max_translation, max_translation + 1))
        dc = int(rng.integers(-max_translation, max_translation + 1))
        candidates.append((base_tl[0] + dr, base_tl[1] + dc))

    # Try candidates
    moved = False
    for tl in candidates:
        if _stamp_object_at(new_env, ob, tl, moat=moat, allow_touch=False):
            moved = True
            break

    if not moved:
        # Restore original placement
        _ = _stamp_object_at(new_env, ob, base_tl, moat=0, allow_touch=True)
        return new_env, False

    # Optional re-label to sanitize (ids will be re-packed; we keep old id by mapping)
    if relabel_after:
        new_env = rebuild_objects_from_grid(new_env)

    # Ensure start/goal remain free
    new_env.grid[new_env.start] = False
    new_env.grid[new_env.goal] = False

    return new_env, True


# ----------------------------- Batch convenience ---------------------------- #

def remove_many_and_move_one(
    env: GridEnvironment,
    remove_ids: Iterable[int],
    move_id: Optional[int],
    *,
    move_kwargs: Optional[dict] = None,
    inplace: bool = False
) -> Tuple[GridEnvironment, bool]:
    """
    Convenience: remove a set of obstacles, then (optionally) move one obstacle.

    Returns (env, moved_flag).
    """
    move_kwargs = move_kwargs or {}
    new_env = remove_obstacles(env, remove_ids, inplace=inplace, relabel_after=False)
    moved_flag = False
    if move_id is not None:
        new_env, moved_flag = move_obstacle(new_env, move_id, inplace=True, **move_kwargs)
    return new_env, moved_flag


# ---------------------------------- Demo ------------------------------------ #

if __name__ == "__main__":
    from envs.generator import generate_environment, _free_bfs_has_path

    rng = np.random.default_rng(7)
    env = generate_environment(H=40, W=40, density=0.20, ensure_status="failure", rng=rng)
    print("Initial objects:", len(env.obstacles),
          "path_exists?", _free_bfs_has_path(env.grid, env.start, env.goal))

    # Try removing one small obstacle (random)
    if env.obstacles:
        small_idx = int(np.argmin([ob.area for ob in env.obstacles])) + 1  # id (1-based)
        env2 = remove_obstacles(env, [small_idx], inplace=False, relabel_after=False)
        print("After remove, path_exists?",
              _free_bfs_has_path(env2.grid, env2.start, env2.goal))

        # Try moving a (different) obstacle
        move_target = min(len(env2.obstacles), small_idx + 1)
        env3, moved = move_obstacle(env2, move_target, max_translation=5, n_candidates=60, moat=1)
        print("Moved?", moved, "path_exists?", _free_bfs_has_path(env3.grid, env3.start, env3.goal))
