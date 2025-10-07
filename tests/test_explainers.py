import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from envs.generator import generate_environment, _free_bfs_has_path
from explainers.baselines import random_ranking, geodesic_line_ranking
from explainers.lime_explainer import LimeExplainer
from explainers.shap_explainer import ShapExplainer
from explainers.cose import COSEExplainer

# --- Tiny planner that matches your explainers' interface ---------------------
class BFSPlanner:
    """Planner.plan(grid,start,goal) -> {'success': bool} using the env BFS check."""
    def plan(self, grid, start, goal):
        ok = _free_bfs_has_path(grid, start, goal)
        return {"success": bool(ok)}

def make_env(seed=0, ensure="failure", H=40, W=40, density=0.2):
    rng = np.random.default_rng(seed)
    return generate_environment(H=H, W=W, density=density, ensure_status=ensure, rng=rng)

# --- Tests --------------------------------------------------------------------
def test_baselines_rank_all_ids():
    env = make_env(seed=1, ensure="any")
    rr = random_ranking(env, random_state=123)
    gg = geodesic_line_ranking(env)
    n = sum(1 for o in env.obstacles if o.coords.size > 0)
    assert len(rr["ranking"]) >= n
    assert len(gg["ranking"]) >= n

def test_lime_runs_and_outputs_scores():
    env = make_env(seed=2, ensure="any")
    planner = BFSPlanner()
    lime = LimeExplainer(num_samples=50, flip_prob=0.3, random_state=7, focus_top_m=min(10, len(env.obstacles)))
    out = lime.explain(env, planner)
    assert "ranking" in out and isinstance(out["ranking"], list)
    # higher score means "more harmful" (responsible for failure)
    assert all(isinstance(i, int) and isinstance(s, float) for (i, s) in out["ranking"])

def test_shap_runs_and_outputs_scores():
    env = make_env(seed=3, ensure="any")
    planner = BFSPlanner()
    shap = ShapExplainer(permutations=30, random_state=0, focus_top_m=min(10, len(env.obstacles)))
    out = shap.explain(env, planner)
    assert "ranking" in out and isinstance(out["ranking"], list)
    assert all(isinstance(i, int) and isinstance(s, float) for (i, s) in out["ranking"])

def test_cose_fixes_failure_with_minimal_set():
    # Start from an enforced-failure map
    env = make_env(seed=4, ensure="failure")
    planner = BFSPlanner()
    assert not _free_bfs_has_path(env.grid, env.start, env.goal)

    # Guide with geodesic LITE ranking
    guide = geodesic_line_ranking(env)["ranking"]
    cose = COSEExplainer(guide='none')  # guide name is metadata; we pass ranking explicitly
    out = cose.explain(env, planner, guide_ranking=guide)

    # COSE should return a set; removing it should yield success
    S = sorted(list(out["cose_set"]))
    assert isinstance(out["cose_set"], set)

    # Apply the removal to confirm success
    grid = env.grid.copy()
    for oid in S:
        grid[env.obj_map == oid] = False
    assert _free_bfs_has_path(grid, env.start, env.goal)

    # Minimality check: no proper subset suffices
    from itertools import combinations
    for k in range(len(S)):
        for subset in combinations(S, k):
            g2 = env.grid.copy()
            for oid in subset:
                g2[env.obj_map == oid] = False
            assert not _free_bfs_has_path(g2, env.start, env.goal)

