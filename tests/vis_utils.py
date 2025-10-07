import os
import numpy as np
import matplotlib.pyplot as plt

# --- Colors/colormap helpers -------------------------------------------------
def _make_colormap(max_id: int):
    # Distinct categorical-ish colormap for object IDs 1..K
    # ID 0 is background, plotted as white.
    cmap = plt.cm.get_cmap("tab20", max(20, max_id + 1))
    return cmap

def render_env(env, ax=None, show_ids=False, title=None):
    """
    Render a GridEnvironment.

    Layers:
      - background (white)
      - obstacles by obj_map ID (colored)
      - start (green star), goal (red star)
    """
    # Prefer env.shape, fall back to grid.shape if shape is not present
    if hasattr(env, "shape"):
        H, W = env.shape
    else:
        H, W = env.grid.shape
    
    if ax is None:
        _, ax = plt.subplots(figsize=(W/5, H/5), dpi=120)

    # Base = white
    rgb = np.ones((H, W, 3), dtype=float)

    # Color obstacles by obj_map id (if present), else by grid mask
    if getattr(env, "obj_map", None) is not None and env.obj_map is not None:
        labels = env.obj_map
        max_id = int(labels.max())
        if max_id > 0:
            cmap = _make_colormap(max_id)
            # map each ID to a distinct color; 0 stays white
            colored = cmap(labels / (max_id + 1))[..., :3]
            # keep white where label == 0
            mask = labels > 0
            rgb[mask] = colored[mask]
        else:
            # fallback to grid mask if no objects
            rgb[env.grid] = 0.2
    else:
        # No obj_map; just fill obstacles (dark gray)
        rgb[env.grid] = 0.2

    ax.imshow(rgb, interpolation="nearest", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])

    # Start/Goal markers
    if getattr(env, "start", None) is not None:
        ax.plot(env.start[1], env.start[0], marker="*", markersize=10, markeredgecolor="k", markerfacecolor="lime", lw=0)
        ax.text(env.start[1]+0.2, env.start[0]-0.2, "S", color="k", fontsize=8)
    if getattr(env, "goal", None) is not None:
        ax.plot(env.goal[1], env.goal[0], marker="*", markersize=10, markeredgecolor="k", markerfacecolor="red",  lw=0)
        ax.text(env.goal[1]+0.2, env.goal[0]-0.2, "G", color="k", fontsize=8)

    # Optional: annotate object IDs
    if show_ids and getattr(env, "obj_map", None) is not None:
        ids = np.unique(env.obj_map)
        ids = ids[ids > 0]
        for oid in ids:
            coords = np.argwhere(env.obj_map == oid)
            if coords.size == 0:
                continue
            cy, cx = coords.mean(axis=0)
            ax.text(cx, cy, str(int(oid)), color="k", fontsize=7, ha="center", va="center")

    if title:
        ax.set_title(title, fontsize=10)

    return ax

def save_side_by_side(env_a, env_b, path, title_a="original", title_b="perturbed", show_ids=False):
    H, W = env_a.shape
    fig, axes = plt.subplots(1, 2, figsize=(max(6, W/4), max(3, H/4)), dpi=140)
    render_env(env_a, ax=axes[0], show_ids=show_ids, title=title_a)
    render_env(env_b, ax=axes[1], show_ids=show_ids, title=title_b)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
