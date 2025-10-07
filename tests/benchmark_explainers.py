import os, sys, time, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from envs.generator import generate_environment, _free_bfs_has_path
from explainers.baselines import geodesic_line_ranking
from explainers.lime_explainer import LimeExplainer
from explainers.shap_explainer import ShapExplainer
from explainers.cose import COSEExplainer

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "benchmark_results.csv")

class CountingPlanner:
    """Wraps BFS reachability and counts calls to .plan(...)"""
    def __init__(self):
        self.calls = 0
    def reset(self):
        self.calls = 0
    def plan(self, grid, start, goal):
        self.calls += 1
        return {"success": _free_bfs_has_path(grid, start, goal)}

def run_once(env_kind: str, seed: int, H=40, W=40, density=0.2,
             lime_samples=200, lime_flip=0.3,
             shap_perms=50, focus_top_m=15):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=H, W=W, density=density, ensure_status=env_kind, rng=rng)

    planner = CountingPlanner()

    results = []

    # --- Baseline: Geodesic (no planner calls) ---
    t0 = time.perf_counter()
    guide = geodesic_line_ranking(env)["ranking"]
    t1 = time.perf_counter()
    results.append({
        "env_kind": env_kind,
        "seed": seed,
        "method": "geodesic_line",
        "wall_time_s": t1 - t0,
        "planner_calls": planner.calls
    })

    # --- LIME ---
    planner.reset()
    lime = LimeExplainer(num_samples=lime_samples, flip_prob=lime_flip,
                         random_state=seed, focus_top_m=min(focus_top_m, len(env.obstacles)))
    t0 = time.perf_counter()
    _ = lime.explain(env, planner)
    t1 = time.perf_counter()
    results.append({
        "env_kind": env_kind,
        "seed": seed,
        "method": "lime",
        "wall_time_s": t1 - t0,
        "planner_calls": planner.calls
    })

    # --- SHAP ---
    planner.reset()
    shap = ShapExplainer(permutations=shap_perms, random_state=seed,
                         focus_top_m=min(focus_top_m, len(env.obstacles)))
    t0 = time.perf_counter()
    _ = shap.explain(env, planner)
    t1 = time.perf_counter()
    results.append({
        "env_kind": env_kind,
        "seed": seed,
        "method": "shap",
        "wall_time_s": t1 - t0,
        "planner_calls": planner.calls
    })

    # --- COSE (guided by Geodesic) ---
    planner.reset()
    cose = COSEExplainer()
    t0 = time.perf_counter()
    _ = cose.explain(env, planner, guide_ranking=guide)
    t1 = time.perf_counter()
    results.append({
        "env_kind": env_kind,
        "seed": seed,
        "method": "cose",
        "wall_time_s": t1 - t0,
        "planner_calls": planner.calls
    })

    return results

def main():
    rows = []
    # A small grid of scenarios; tweak freely
    cases = [
        ("any",  0, 40, 40, 0.18),
        ("any",  1, 50, 50, 0.18),
        ("failure", 2, 40, 40, 0.20),
        ("failure", 3, 50, 50, 0.22),
    ]
    for env_kind, seed, H, W, density in cases:
        rows.extend(run_once(env_kind, seed, H, W, density))

    # Print table
    print(f"{'env':8} {'seed':4} {'method':12} {'time[s]':>9} {'planner_calls':>15}")
    for r in rows:
        print(f"{r['env_kind']:8} {r['seed']:4d} {r['method']:12} {r['wall_time_s']:9.4f} {r['planner_calls']:15d}")

    # Save CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env_kind","seed","method","wall_time_s","planner_calls"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {CSV_PATH}")

if __name__ == "__main__":
    main()
