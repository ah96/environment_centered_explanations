import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.generator import generate_environment
from planners import PLANNERS
from tests.vis_utils import render_env
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

def visualize_path(env, path, title, fname):
    fig, ax = plt.subplots(figsize=(env.grid.shape[1]/5, env.grid.shape[0]/5), dpi=120)
    render_env(env, ax=ax, title=title)
    if path:
        rr, cc = zip(*path)
        ax.plot(cc, rr, color="lime", lw=2, alpha=0.8)
    out_path = os.path.join(OUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def test_all_planners(seed=42):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=50, W=50, density=0.18, ensure_status="any", rng=rng)

    for name, cls in PLANNERS.items():
        planner = cls(connectivity=8)
        res = planner.plan(env.grid, env.start, env.goal)
        title = f"{name}: {'success' if res['success'] else 'fail'}"
        path = res['path'] if res['success'] else None
        fname = f"{name}_path_seed{seed}.png"
        visualize_path(env, path, title, fname)

if __name__ == "__main__":
    test_all_planners()
