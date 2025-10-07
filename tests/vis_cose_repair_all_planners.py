import os, sys, math, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from envs.generator import generate_environment, _free_bfs_has_path
from explainers.baselines import geodesic_line_ranking
from explainers.cose import COSEExplainer
from planners import PLANNERS
from tests.vis_utils import render_env

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

def geom_length(path):
    if not path or len(path) < 2: return 0.0
    g = 0.0
    for (r0,c0),(r1,c1) in zip(path[:-1], path[1:]):
        g += math.hypot(r1-r0, c1-c0)
    return g

def overlay_ids(ax, env, ids, alpha=0.55):
    if not ids: return
    mask = np.isin(env.obj_map, np.array(ids, dtype=int))
    H, W = env.grid.shape
    rgba = np.zeros((H, W, 4), float)
    rgba[...,0] = 1.0
    rgba[...,3] = 0.0
    rgba[mask,3] = alpha
    ax.imshow(rgba, origin="upper", interpolation="nearest")

def main(seed=11, H=50, W=50, density=0.22):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=H, W=W, density=density, ensure_status="failure", rng=rng)
    assert not _free_bfs_has_path(env.grid, env.start, env.goal)

    guide = geodesic_line_ranking(env)["ranking"]
    class PlannerLite:
        def plan(self, grid, start, goal):
            return {"success": _free_bfs_has_path(grid, start, goal)}

    cose = COSEExplainer()
    out = cose.explain(env, PlannerLite(), guide_ranking=guide)
    cose_ids = sorted(list(out["cose_set"]))

    # repaired grid
    G_after = env.grid.copy()
    for oid in cose_ids:
        G_after[env.obj_map == oid] = False
    assert _free_bfs_has_path(G_after, env.start, env.goal)

    # figure
    H0, W0 = env.grid.shape
    fig = plt.figure(figsize=(max(10, W0/3), max(6, H0/3)), dpi=140)
    gs = fig.add_gridspec(2, 3, width_ratios=[1,1,1], height_ratios=[1,1], wspace=0.15, hspace=0.2)

    # Left-top: original + COSE overlay
    ax0 = fig.add_subplot(gs[0, 0])
    render_env(env, ax=ax0, show_ids=False, title=f"Original (blocked)\nCOSE set: {cose_ids}")
    overlay_ids(ax0, env, cose_ids, alpha=0.55)

    # Right-top: repaired base (no path) as legend panel
    import copy as _copy
    env_after_base = _copy.copy(env)     # keep class (and .shape)
    env_after_base.grid = G_after
    ax1 = fig.add_subplot(gs[0, 1])
    render_env(env_after_base, ax=ax1, show_ids=False, title="Repaired (no path overlay)")

    # Now plot planners (5 slots: a_star, dijkstra, bfs, dfs, theta_star)
    slots = [(0,2),(1,0),(1,1),(1,2)]
    names = ["a_star","dijkstra","bfs","dfs","theta_star"]
    panels = [fig.add_subplot(gs[r, c]) for (r,c) in [(0,2),(1,0),(1,1),(1,2)]]
    # If you prefer a fixed layout, adjust as needed.

    # We’ll plot all 5; reusing or extending panels list
    while len(panels) < len(names):
        panels.append(fig.add_subplot(gs[1,2]))

    for ax, name in zip(panels, names):
        planner = PLANNERS[name](connectivity=8)
        res = planner.plan(G_after, env.start, env.goal)
        title = f"{name}: {'success' if res['success'] else 'fail'}"
        render_env(env_after_base, ax=ax, show_ids=False, title=title)
        if res['success']:
            rr, cc = zip(*res['path'])
            ax.plot(cc, rr, lw=2)
            # annotate metrics
            gl = geom_length(res['path'])
            ax.text(0.02, 0.02, f"len={gl:.1f}, hops={len(res['path'])-1}",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))

    out_path = os.path.join(OUT_DIR, f"cose_repair_all_planners_seed{seed}.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
