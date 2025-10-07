import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.generator import generate_environment, _free_bfs_has_path
from explainers.baselines import geodesic_line_ranking
from explainers.cose import COSEExplainer
from planners import PLANNERS

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

def geom_length(path):
    if not path or len(path) < 2: return 0.0
    g = 0.0
    for (r0,c0),(r1,c1) in zip(path[:-1], path[1:]):
        g += math.hypot(r1-r0, c1-c0)
    return g

def main(seed=11, H=50, W=50, density=0.22):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=H, W=W, density=density, ensure_status="failure", rng=rng)
    assert not _free_bfs_has_path(env.grid, env.start, env.goal)

    # COSE set
    guide = geodesic_line_ranking(env)["ranking"]
    class PlannerLite:
        def plan(self, grid, start, goal):
            return {"success": _free_bfs_has_path(grid, start, goal)}
    cose = COSEExplainer()
    out = cose.explain(env, PlannerLite(), guide_ranking=guide)
    cose_ids = sorted(list(out["cose_set"]))
    print("COSE set:", cose_ids)

    # Apply removal
    G_after = env.grid.copy()
    for oid in cose_ids:
        G_after[env.obj_map == oid] = False
    assert _free_bfs_has_path(G_after, env.start, env.goal)

    # Run all planners on the repaired grid
    rows = []
    for name, cls in PLANNERS.items():
        planner = cls(connectivity=8)
        res = planner.plan(G_after, env.start, env.goal)
        ok = int(bool(res["success"]))
        hops = len(res["path"])-1 if res["path"] else 0
        gl = geom_length(res["path"]) if res["path"] else 0.0
        rows.append((name, ok, hops, gl))
    for r in rows:
        print(f"{r[0]:11} success={r[1]} hops={r[2]:4d} geom={r[3]:6.2f}")

if __name__ == "__main__":
    main()
