import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from envs.generator import generate_environment, _free_bfs_has_path
from envs.affordances import remove_obstacles, move_obstacle
from tests.vis_utils import render_env, save_side_by_side

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")

def pick_smallest_id(env):
    # chooses smallest-area active obstacle id (1-based)
    areas = []
    for idx, ob in enumerate(env.obstacles, start=1):
        area = getattr(ob, "area", (len(ob.coords) if hasattr(ob, "coords") else 0))
        areas.append((area, idx))
    areas = [(a, i) for (a, i) in areas if a > 0]
    if not areas:
        return None
    areas.sort()
    return areas[0][1]

def visualize_random_any(seed=0, H=40, W=40, density=0.18):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=H, W=W, density=density, ensure_status="any", rng=rng)
    # Original
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W/5, H/5), dpi=140)
    render_env(env, show_ids=True, title=f"ANY env (seed={seed})")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"any_env_seed{seed}.png"), bbox_inches="tight")
    plt.close(fig)

def visualize_failure_then_remove(seed=1, H=40, W=40, density=0.2):
    rng = np.random.default_rng(seed)
    env_fail = generate_environment(H=H, W=W, density=density, ensure_status="failure", rng=rng)
    assert not _free_bfs_has_path(env_fail.grid, env_fail.start, env_fail.goal)
    # remove smallest
    oid = pick_smallest_id(env_fail)
    if oid is None:
        return
    env_removed = remove_obstacles(env_fail, [oid], inplace=False, relabel_after=False)
    save_side_by_side(env_fail, env_removed,
                      os.path.join(OUT_DIR, f"failure_remove_id{oid}_seed{seed}.png"),
                      title_a="failure (no path)", title_b=f"after remove id={oid}",
                      show_ids=True)

def visualize_any_then_move(seed=2, H=50, W=50, density=0.18):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=H, W=W, density=density, ensure_status="any", rng=rng)
    # choose a random valid id
    if not env.obstacles:
        return
    oid = int(rng.integers(1, len(env.obstacles)+1))
    env_moved, moved = move_obstacle(env, oid, max_translation=8, n_candidates=200, moat=1)
    title_b = f"after move id={oid}" + (" (moved)" if moved else " (no candidate)")
    save_side_by_side(env, env_moved,
                      os.path.join(OUT_DIR, f"any_move_id{oid}_seed{seed}.png"),
                      title_a="original (any)", title_b=title_b,
                      show_ids=True)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    visualize_random_any(seed=0)
    visualize_failure_then_remove(seed=1)
    visualize_any_then_move(seed=2)
    print(f"Saved images to: {OUT_DIR}")
