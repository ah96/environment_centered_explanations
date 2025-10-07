import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from envs.generator import generate_environment, _free_bfs_has_path
from explainers.baselines import geodesic_line_ranking
from explainers.lime_explainer import LimeExplainer
from explainers.shap_explainer import ShapExplainer
from explainers.cose import COSEExplainer

from tests.vis_utils import render_env, save_side_by_side

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

def _overlay_ids(ax, env, ids, label, alpha=0.45):
    """Overlay semi-transparent highlight on the cells of the given obstacle IDs."""
    if ids is None or len(ids) == 0:
        return
    mask = np.isin(env.obj_map, np.array(list(ids), dtype=int))
    H, W = env.shape
    overlay = np.zeros((H, W, 4), dtype=float)  # RGBA
    overlay[..., 0] = 1.0  # red channel
    overlay[..., 3] = 0.0  # alpha
    overlay[mask, 3] = alpha
    ax.imshow(overlay, origin="upper", interpolation="nearest")
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.5))

def visualize_topk_overlays(seed=10, H=50, W=50, density=0.18, k=8):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=H, W=W, density=density, ensure_status="any", rng=rng)

    # tiny planner
    class Planner:
        def plan(self, grid, start, goal):
            return {"success": _free_bfs_has_path(grid, start, goal)}

    planner = Planner()

    # LIME
    lime = LimeExplainer(num_samples=200, flip_prob=0.3, random_state=seed,
                         focus_top_m=min(20, len(env.obstacles)))
    lime_rank = lime.explain(env, planner)["ranking"]
    lime_ids = [i for i, s in lime_rank[:k]]

    # SHAP
    shap = ShapExplainer(permutations=50, random_state=seed,
                         focus_top_m=min(20, len(env.obstacles)))
    shap_rank = shap.explain(env, planner)["ranking"]
    shap_ids = [i for i, s in shap_rank[:k]]

    # Plot overlays
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
    render_env(env, ax=axes[0], show_ids=False, title=f"LIME top-{k}")
    _overlay_ids(axes[0], env, lime_ids, label="LIME")

    render_env(env, ax=axes[1], show_ids=False, title=f"SHAP top-{k}")
    _overlay_ids(axes[1], env, shap_ids, label="SHAP")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"topk_overlays_seed{seed}_k{k}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def visualize_cose_before_after(seed=11, H=40, W=40, density=0.20):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=H, W=W, density=density, ensure_status="failure", rng=rng)
    assert not _free_bfs_has_path(env.grid, env.start, env.goal)

    # guide -> COSE
    guide = geodesic_line_ranking(env)["ranking"]
    class Planner:
        def plan(self, grid, start, goal):
            return {"success": _free_bfs_has_path(grid, start, goal)}
    planner = Planner()
    cose = COSEExplainer()
    out = cose.explain(env, planner, guide_ranking=guide)
    cose_ids = sorted(list(out["cose_set"]))

    # Build a "repaired" env grid for the after-view
    G_after = env.grid.copy()
    for oid in cose_ids:
        G_after[env.obj_map == oid] = False
    success = _free_bfs_has_path(G_after, env.start, env.goal)

    # Draw side-by-side
    # Left: original with COSE red overlay; Right: after-removal with caption
    import copy
    env_after = copy.copy(env)  # preserve class & properties (e.g., .shape)
    env_after.grid = G_after

    H0, W0 = env.shape
    fig, axes = plt.subplots(1, 2, figsize=(max(6, W0/4), max(3, H0/4)), dpi=140)
    render_env(env, ax=axes[0], show_ids=False, title=f"COSE set: {cose_ids}")
    _overlay_ids(axes[0], env, cose_ids, label="COSE remove", alpha=0.55)

    title_b = "after COSE removal — "
    title_b += "PATH FOUND" if success else "still blocked"
    render_env(env_after, ax=axes[1], show_ids=False, title=title_b)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"cose_before_after_seed{seed}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    visualize_topk_overlays(seed=10, k=8)
    visualize_cose_before_after(seed=11)
    print(f"Images saved under: {OUT_DIR}")
