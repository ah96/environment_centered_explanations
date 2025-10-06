#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generator.py
------------
Lightweight, high-performance 2D occupancy-grid environment generator with
object-level obstacles (connected components) intended for planner-agnostic
explanation research (LIME, SHAP, COSE).

Key design goals:
- Object-level semantics: every obstacle is a *single connected component*.
- Clean feature mapping: each object has a stable integer id and a pixel list.
- Controllable difficulty: random clutter + optional "force failure" via
  minimal path cuts; optional "near-failure" corridor shaping.
- Efficient geometry: pure NumPy + SciPy ndimage kernels; no heavyweight sims.
- Reproducibility: explicit np.random.Generator with seed.

Dependencies:
    numpy
    scipy.ndimage   (for connected-component labeling & binary dilation)

Usage (quick smoke test):
    python3 generator.py

Author: RAL Paper Toolkit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np

try:
    # SciPy is standard in scientific Python stacks; used for labeling & dilation.
    from scipy.ndimage import label as cc_label
    from scipy.ndimage import binary_dilation
except Exception as e:
    raise ImportError(
        "scipy.ndimage is required. Install with: pip install scipy"
    ) from e


# ------------------------------- Data classes ------------------------------- #

@dataclass
class Obstacle:
    """A single connected obstacle (object) represented by its pixel coordinates."""
    id: int
    # Integer pixel coordinates as (row, col) pairs; shape (N, 2).
    coords: np.ndarray
    # Bounding box (inclusive-exclusive): (r0, r1, c0, c1)
    bbox: Tuple[int, int, int, int]

    @property
    def area(self) -> int:
        return int(self.coords.shape[0])


@dataclass
class GridEnvironment:
    """Occupancy-grid world with object-level obstacles."""
    grid: np.ndarray            # (H, W) bool array: True = occupied, False = free
    obj_map: np.ndarray         # (H, W) int32 array: 0=free; >0 = obstacle id
    obstacles: List[Obstacle]   # all obstacles (ids are 1..len(obstacles))
    start: Tuple[int, int]
    goal: Tuple[int, int]
    settings: Dict              # record of generator settings used (for provenance)
    rng: np.random.Generator

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    @property
    def H(self) -> int:
        return self.grid.shape[0]

    @property
    def W(self) -> int:
        return self.grid.shape[1]


# ------------------------------ Utility helpers ----------------------------- #

# 8-connected neighborhood deltas
DELTAS_8 = np.array([
    (-1, -1), (-1, 0), (-1, +1),
    ( 0, -1),          ( 0, +1),
    (+1, -1), (+1, 0), (+1, +1),
], dtype=np.int8)

# 4-connected neighborhood deltas
DELTAS_4 = np.array([
    (-1, 0), (1, 0), (0, -1), (0, 1)
], dtype=np.int8)


def _in_bounds(r: int, c: int, H: int, W: int) -> bool:
    return (0 <= r < H) and (0 <= c < W)


def _free_bfs_has_path(grid: np.ndarray,
                       start: Tuple[int, int],
                       goal: Tuple[int, int],
                       connectivity: int = 8) -> bool:
    """
    Fast boolean reachability test on the free cells of 'grid'.

    grid: bool (True=occupied/blocked)
    returns: True if a free-space path exists from start to goal.
    """
    H, W = grid.shape
    sr, sc = start
    gr, gc = goal

    if grid[sr, sc] or grid[gr, gc]:
        return False  # start or goal is blocked

    visited = np.zeros_like(grid, dtype=bool)
    q = [(sr, sc)]
    visited[sr, sc] = True

    deltas = DELTAS_8 if connectivity == 8 else DELTAS_4

    head = 0  # manual queue for speed
    while head < len(q):
        r, c = q[head]
        head += 1
        if (r, c) == (gr, gc):
            return True
        for dr, dc in deltas:
            nr, nc = r + int(dr), c + int(dc)
            if 0 <= nr < H and 0 <= nc < W and (not visited[nr, nc]) and (not grid[nr, nc]):
                visited[nr, nc] = True
                q.append((nr, nc))
    return False


def _free_bfs_shortest_path(grid: np.ndarray,
                            start: Tuple[int, int],
                            goal: Tuple[int, int],
                            connectivity: int = 8) -> Optional[List[Tuple[int, int]]]:
    """
    Returns *one* shortest free-space path as a list of (r,c) or None if not reachable.
    """
    H, W = grid.shape
    sr, sc = start
    gr, gc = goal
    if grid[sr, sc] or grid[gr, gc]:
        return None

    deltas = DELTAS_8 if connectivity == 8 else DELTAS_4

    visited = np.full((H, W), False, dtype=bool)
    parent_r = np.full((H, W), -1, dtype=np.int32)
    parent_c = np.full((H, W), -1, dtype=np.int32)

    q = [(sr, sc)]
    visited[sr, sc] = True

    head = 0
    found = False
    while head < len(q):
        r, c = q[head]
        head += 1
        if (r, c) == (gr, gc):
            found = True
            break
        for dr, dc in deltas:
            nr, nc = r + int(dr), c + int(dc)
            if 0 <= nr < H and 0 <= nc < W and (not visited[nr, nc]) and (not grid[nr, nc]):
                visited[nr, nc] = True
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                q.append((nr, nc))

    if not found:
        return None

    # Reconstruct path from goal back to start
    path = []
    r, c = gr, gc
    while (r, c) != (sr, sc):
        path.append((r, c))
        pr, pc = parent_r[r, c], parent_c[r, c]
        r, c = int(pr), int(pc)
    path.append((sr, sc))
    path.reverse()
    return path


def _bbox_from_coords(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max()) + 1
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max()) + 1
    return (r0, r1, c0, c1)


def _label_to_obstacles(grid: np.ndarray) -> Tuple[np.ndarray, List[Obstacle]]:
    """
    Given a binary obstacle grid, compute connected components and return:
    - obj_map: int32 map with labels (0=free, 1..K=objects)
    - obstacles: list of Obstacle with stable ids (1..K)
    """
    structure = np.ones((3, 3), dtype=np.uint8)
    obj_map, num = cc_label(grid.astype(np.uint8), structure=structure)
    obstacles: List[Obstacle] = []
    for oid in range(1, num + 1):
        coords = np.column_stack(np.where(obj_map == oid))  # (N, 2): (r,c)
        obstacles.append(Obstacle(id=oid, coords=coords, bbox=_bbox_from_coords(coords)))
    return obj_map.astype(np.int32), obstacles


def _dilate_bool(img: np.ndarray, iters: int = 1) -> np.ndarray:
    if iters <= 0:
        return img
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(img, structure=structure, iterations=int(iters))


# -------------------------- Shape / stamping helpers ------------------------ #

def _stamp_mask(grid: np.ndarray,
                top_left: Tuple[int, int],
                mask: np.ndarray,
                moat: int,
                allow_touch: bool = False) -> bool:
    """
    Attempts to stamp a binary 'mask' (True=occupied) onto 'grid' with its
    top-left corner at 'top_left', enforcing a moat (cell-distance padding)
    around the *new* object unless allow_touch=True.

    Returns True if stamped successfully; False otherwise.
    """
    H, W = grid.shape
    mr, mc = mask.shape
    r0, c0 = top_left
    r1, c1 = r0 + mr, c0 + mc
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        return False

    # Target slice
    target = grid[r0:r1, c0:c1]
    if (target & mask).any():
        return False  # overlaps existing obstacles

    if not allow_touch and moat > 0:
        # Enforce 1+ cell free ring around the new object by checking dilation
        # of the *candidate* mask against the existing grid (outside the mask).
        # Build a slightly larger ROI to check boundary touches too.
        dilated_new = _dilate_bool(mask, iters=moat)

        # Padded ROI on the *grid*
        rpad0 = max(0, r0 - moat)
        cpad0 = max(0, c0 - moat)
        rpad1 = min(H, r1 + moat)
        cpad1 = min(W, c1 + moat)
        grid_roi = grid[rpad0:rpad1, cpad0:cpad1]

        # Canvas (same shape as grid_roi) where we place the dilated_new mask
        canvas_h = rpad1 - rpad0
        canvas_w = cpad1 - cpad0
        ring_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)

        # Where to insert dilated_new into ring_canvas
        ins_r0 = max(0, r0 - rpad0)
        ins_c0 = max(0, c0 - cpad0)
        ins_r1 = min(canvas_h, ins_r0 + dilated_new.shape[0])
        ins_c1 = min(canvas_w, ins_c0 + dilated_new.shape[1])

        # Corresponding slice on dilated_new (clamped; avoids negative indices)
        src_r0 = max(0, -(r0 - rpad0))
        src_c0 = max(0, -(c0 - cpad0))
        src_r1 = src_r0 + (ins_r1 - ins_r0)
        src_c1 = src_c0 + (ins_c1 - ins_c0)

        ring_canvas[ins_r0:ins_r1, ins_c0:ins_c1] = dilated_new[src_r0:src_r1, src_c0:src_c1]

        # If the *dilated ring* touches any existing obstacle, reject placement
        if (grid_roi & ring_canvas).any():
            return False

    # OK to stamp
    target |= mask
    return True


def _random_rectangle_mask(rng: np.random.Generator,
                           min_h: int, max_h: int,
                           min_w: int, max_w: int) -> np.ndarray:
    h = int(rng.integers(min_h, max_h + 1))
    w = int(rng.integers(min_w, max_w + 1))
    h = max(1, h)
    w = max(1, w)
    return np.ones((h, w), dtype=bool)


def _random_blob_mask(rng: np.random.Generator,
                      min_cells: int, max_cells: int,
                      growth_bias_4_conn: float = 0.6) -> np.ndarray:
    """
    Generate a small connected 'polyomino'-like blob by random growth.
    Returns a tight bounding-box mask (bool) of the blob.

    growth_bias_4_conn: bias to pick 4-connected neighbors vs diagonals to
                        reduce skinny diagonal chains.
    """
    n = int(rng.integers(min_cells, max_cells + 1))
    n = max(1, n)
    # We grow on a temporary canvas; the final bbox mask is returned.
    # Canvas size heuristic: enough room to grow without hitting borders.
    side = int(np.ceil(np.sqrt(n))) + 4
    canvas = np.zeros((side, side), dtype=bool)
    # Start near center
    r = c = side // 2
    canvas[r, c] = True
    coords = [(r, c)]

    # Precompute neighbor order with bias
    deltas = list(DELTAS_4) + list(DELTAS_8)
    for _ in range(n - 1):
        # pick a random existing cell to grow from
        base_r, base_c = coords[rng.integers(0, len(coords))]
        # explore neighbors with a bias
        if rng.random() < growth_bias_4_conn:
            neighs = list(DELTAS_4)
        else:
            neighs = list(DELTAS_8)
        rng.shuffle(neighs)
        grown = False
        for dr, dc in neighs:
            nr, nc = base_r + int(dr), base_c + int(dc)
            if 0 <= nr < side and 0 <= nc < side and (not canvas[nr, nc]):
                canvas[nr, nc] = True
                coords.append((nr, nc))
                grown = True
                break
        if not grown:
            # fallback: pick any free cell adjacent to the current blob
            # (simple, robust)
            free = np.argwhere(~canvas)
            rng.shuffle(free)
            for fr, fc in free:
                # neighbor to blob?
                if np.any(canvas[max(0, fr-1):fr+2, max(0, fc-1):fc+2]):
                    canvas[fr, fc] = True
                    coords.append((fr, fc))
                    break

    # Tight bbox
    ys, xs = np.where(canvas)
    r0, r1 = ys.min(), ys.max() + 1
    c0, c1 = xs.min(), xs.max() + 1
    return canvas[r0:r1, c0:c1]


# ------------------------------- Core generator ----------------------------- #

def generate_environment(
    H: int = 48,
    W: int = 48,
    *,
    density: Optional[float] = 0.18,
    n_objects: Optional[int] = None,
    start: Tuple[int, int] = (0, 0),
    goal: Optional[Tuple[int, int]] = None,
    moat: int = 1,
    shape_probs: Dict[str, float] = None,
    rect_size: Tuple[Tuple[int, int], Tuple[int, int]] = ((2, 5), (2, 7)),  # (min_h,max_h),(min_w,max_w)
    blob_cells: Tuple[int, int] = (4, 16),   # min,max pixels in blob
    ensure_status: str = "any",              # "any" | "failure" | "near-failure"
    connectivity: int = 8,
    rng: Optional[np.random.Generator] = None,
    max_place_tries: int = 5000,
) -> GridEnvironment:
    """
    Create an occupancy-grid with object-level obstacles.

    Strategy:
      1) Randomly place object masks (rectangles + blobs) until hitting either
         target density or object count, enforcing a 'moat' separation.
      2) Optionally enforce 'failure' by cutting the shortest free path with a
         tiny object (kept disjoint from others).
      3) Optionally enforce 'near-failure' by nudging clutter around a corridor.

    ensure_status:
        "any"          : no guarantee about path existence.
        "failure"      : guarantees no free path from start to goal by cutting it.
        "near-failure" : encourages a narrow corridor but keeps a path.

    Returns a GridEnvironment with fully populated fields.
    """
    assert connectivity in (4, 8)
    if goal is None:
        goal = (H - 1, W - 1)

    if shape_probs is None:
        shape_probs = {"rect": 0.6, "blob": 0.4}
    # Normalize shape probabilities
    tot = float(sum(shape_probs.values()))
    shape_keys = list(shape_probs.keys())
    shape_p = np.array([shape_probs[k] / tot for k in shape_keys], dtype=float)

    rng = rng or np.random.default_rng()

    grid = np.zeros((H, W), dtype=bool)
    settings = dict(
        H=H, W=W, density=density, n_objects=n_objects, start=start, goal=goal,
        moat=moat, shape_probs=shape_probs, rect_size=rect_size,
        blob_cells=blob_cells, ensure_status=ensure_status,
        connectivity=connectivity, max_place_tries=max_place_tries,
        seed=int(rng.integers(0, 2**31-1)),
    )

    # Helper to place one object mask
    def try_place_one(mask: np.ndarray, tries: int = 100) -> bool:
        mr, mc = mask.shape
        for _ in range(tries):
            r0 = int(rng.integers(0, H - mr + 1))
            c0 = int(rng.integers(0, W - mc + 1))
            if _stamp_mask(grid, (r0, c0), mask, moat=moat, allow_touch=False):
                return True
        return False

    # Determine placement budget
    target_cells = None
    if density is not None:
        density = float(np.clip(density, 0.0, 0.9))
        target_cells = int(round(density * H * W))
    placed_objects = 0
    tries = 0

    # -------------------- 1) Random object placement -------------------- #
    while tries < max_place_tries:
        tries += 1
        # Stop if density target met
        if target_cells is not None and int(grid.sum()) >= target_cells:
            break
        # Or if object count target met
        if n_objects is not None and placed_objects >= int(n_objects):
            break

        # Sample shape type
        shape = shape_keys[rng.choice(len(shape_keys), p=shape_p)]
        if shape == "rect":
            (min_h, max_h), (min_w, max_w) = rect_size
            mask = _random_rectangle_mask(rng, min_h, max_h, min_w, max_w)
        else:
            mask = _random_blob_mask(rng, blob_cells[0], blob_cells[1])

        if try_place_one(mask, tries=30):
            placed_objects += 1

    # Ensure start/goal free
    grid[start] = False
    grid[goal] = False

    # -------------------- 2) Enforce difficulty if requested ------------- #
    if ensure_status == "failure":
        # If a path exists, cut it with a minimal new object not touching others.
        # We allow 256 attempts to create a tiny blocking bar that does not merge.
        for _ in range(256):
            path = _free_bfs_shortest_path(grid, start, goal, connectivity)
            if path is None:
                break  # already no path
            # Pick a middle segment to block
            mid_idx = int(len(path) // 2)
            pr, pc = path[mid_idx]

            # Try to place a 1x3 or 3x1 bar perpendicular-ish to the local motion
            # Fallback to single cell if bars fail repeatedly.
            for _bar_try in range(8):
                # Randomly choose orientation
                if rng.random() < 0.5:
                    # vertical bar
                    bar_h, bar_w = rng.integers(2, 4), 1
                else:
                    # horizontal bar
                    bar_h, bar_w = 1, rng.integers(2, 4)
                r0 = int(np.clip(pr - bar_h // 2, 0, H - bar_h))
                c0 = int(np.clip(pc - bar_w // 2, 0, W - bar_w))
                mask = np.ones((bar_h, bar_w), dtype=bool)

                # For this *cut* object, we still prefer not touching others to
                # keep it a distinct obstacle; allow_touch=False with moat=1 is fine.
                if _stamp_mask(grid, (r0, c0), mask, moat=1, allow_touch=False):
                    break
            else:
                # As a last resort, try a single cell plug
                if not _stamp_mask(grid, (pr, pc), np.ones((1, 1), dtype=bool), moat=1, allow_touch=False):
                    # if that even fails (rare), perturb mid point slightly
                    for _j in range(10):
                        rr = int(np.clip(pr + int(rng.integers(-1, 2)), 0, H - 1))
                        cc = int(np.clip(pc + int(rng.integers(-1, 2)), 0, W - 1))
                        if _stamp_mask(grid, (rr, cc), np.ones((1, 1), dtype=bool), moat=1, allow_touch=False):
                            break

            # Loop again; if still path exists, we’ll add another tiny cut object
            # until no path remains or attempts are exhausted.

    elif ensure_status == "near-failure":
        # Encourage a single narrow corridor by adding side clutter near a shortest path
        path = _free_bfs_shortest_path(grid, start, goal, connectivity)
        if path is not None and len(path) > 4:
            # Place small 1x1 pebbles adjacent to random path cells, not on them.
            added = 0
            for _ in range(64):
                pr, pc = path[int(rng.integers(1, len(path) - 1))]
                # Choose a side neighbor not on path
                neighs = DELTAS_4.copy()
                rng.shuffle(neighs)
                placed = False
                for dr, dc in neighs:
                    rr, cc = pr + int(dr), pc + int(dc)
                    if 0 <= rr < H and 0 <= cc < W and (not grid[rr, cc]):
                        if _stamp_mask(grid, (rr, cc), np.ones((1, 1), dtype=bool), moat=1, allow_touch=False):
                            added += 1
                            placed = True
                            break
                if placed and added >= 6:
                    break  # a bit of crowding, but keep the corridor open

        # Make sure we didn't accidentally block
        if not _free_bfs_has_path(grid, start, goal, connectivity):
            # If blocked, remove a random small pebble by clearing a 1x1 near path end
            # until path exists; this is rare with the above.
            for _ in range(128):
                r = int(rng.integers(0, H))
                c = int(rng.integers(0, W))
                if grid[r, c]:
                    old = grid[r, c]
                    grid[r, c] = False
                    if _free_bfs_has_path(grid, start, goal, connectivity):
                        break
                    else:
                        grid[r, c] = old

    # Ensure start/goal are free (again) in case clutter touched them
    grid[start] = False
    grid[goal] = False

    # Label connected components and build object list + object map
    obj_map, obstacles = _label_to_obstacles(grid)

    # Re-pack ids to 1..K in case label returns holes (it doesn't, but be robust)
    # (Already 1..num from cc_label.)

    return GridEnvironment(
        grid=grid,
        obj_map=obj_map,
        obstacles=obstacles,
        start=start,
        goal=goal,
        settings=settings,
        rng=rng
    )


# ------------------------------ Maintenance ops ----------------------------- #

def rebuild_objects_from_grid(env: GridEnvironment) -> GridEnvironment:
    """
    Recompute connected components from env.grid and return a *new* environment
    with updated obj_map + obstacles. Preserves start/goal/settings/rng.
    """
    obj_map, obstacles = _label_to_obstacles(env.grid)
    return GridEnvironment(
        grid=env.grid.copy(),
        obj_map=obj_map,
        obstacles=obstacles,
        start=env.start,
        goal=env.goal,
        settings=dict(env.settings),
        rng=env.rng
    )


# ------------------------------ Tiny helper ----------------------------- #

def active_obstacles(env: GridEnvironment):
    """Yield obstacles with non-empty coords (after removals)."""
    return [ob for ob in env.obstacles if ob.coords.size > 0]


# ---------------------------------- Demo ------------------------------------ #

if __name__ == "__main__":
    rng = np.random.default_rng(123)
    env = generate_environment(
        H=48, W=64, density=0.20, start=(0, 0), goal=(47, 63),
        moat=1, ensure_status="failure", rng=rng
    )

    print("Environment:", env.shape, "Start:", env.start, "Goal:", env.goal)
    print("#Obstacle pixels:", int(env.grid.sum()))
    print("#Objects:", len(env.obstacles))
    # Quick path check
    reachable = _free_bfs_has_path(env.grid, env.start, env.goal, connectivity=8)
    print("Path exists?", reachable)
    # Object sizes stats
    sizes = [ob.area for ob in env.obstacles]
    if sizes:
        print("Obstacle area (min/mean/max):",
              np.min(sizes), int(np.mean(sizes)), np.max(sizes))
