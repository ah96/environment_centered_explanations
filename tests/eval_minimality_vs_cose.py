import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.generator import generate_environment
from explainers.baselines import geodesic_line_ranking
from explainers.cose import COSEExplainer
from eval.ilp_minimality import exact_minimal_set
from eval.metrics import jaccard

def main(seed=2):
    rng = np.random.default_rng(seed)
    env = generate_environment(H=25, W=25, density=0.2, ensure_status="failure", rng=rng)

    guide = geodesic_line_ranking(env)["ranking"]
    class BFSPlanner:
        def plan(self, grid, start, goal):
            from envs.generator import _free_bfs_has_path
            return {"success": _free_bfs_has_path(grid, start, goal)}

    cose = COSEExplainer()
    cose_out = cose.explain(env, BFSPlanner(), guide_ranking=guide)
    ilp_out = exact_minimal_set(env, time_limit=30)

    print("COSE set:", cose_out["cose_set"])
    print("ILP set:", ilp_out["set"])
    jac = jaccard(cose_out["cose_set"], ilp_out["set"])
    print(f"Jaccard(COSE,ILP)={jac:.2f}  |  ILP time={ilp_out['time_sec']:.2f}s  status={ilp_out['status']}")

if __name__ == "__main__":
    main()
