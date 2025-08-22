#!/usr/bin/env python3
import importlib, sys, traceback, numpy as np, os
from pathlib import Path

# --- Ensure the repo root is on sys.path ---
ROOT = Path(__file__).resolve().parent.parent  # repo root = parent of scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK = "\x1b[92mOK\x1b[0m"
BAD = "\x1b[91mERR\x1b[0m"

def check(name, fn):
    try:
        fn()
        print(f"[{OK}] {name}")
    except Exception as e:
        print(f"[{BAD}] {name}: {e}")
        traceback.print_exc()
        sys.exit(1)

def test_envs():
    gen = importlib.import_module("envs.generator")
    env = gen.generate_environment(H=20, W=20, density=0.30)
    assert hasattr(env, "grid") and hasattr(env, "obj_map")
    assert env.grid.shape == env.obj_map.shape

def test_planner():
    A = importlib.import_module("planners.a_star").AStarPlanner(connectivity=8)
    from envs.generator import generate_environment
    env = generate_environment(H=20, W=20, density=0.30)
    res = A.plan(env.grid, env.start, env.goal)
    assert isinstance(res, dict) and "success" in res

def test_explainers():
    from planners.a_star import AStarPlanner
    from envs.generator import generate_environment
    shap = importlib.import_module("explainers.shap_explainer").ShapExplainer(permutations=10, random_state=0)
    lime = importlib.import_module("explainers.lime_explainer").LimeExplainer(num_samples=50, random_state=0)
    cose = importlib.import_module("explainers.cose").COSEExplainer(guide="shap")
    A = AStarPlanner(connectivity=8)
    # find a failing env quickly
    for _ in range(60):
        env = generate_environment(H=20, W=20, density=0.35)
        if not A.plan(env.grid, env.start, env.goal)["success"]:
            break
    else:
        raise RuntimeError("Could not find failing env; increase density")
    r_shap = shap.explain(env, A)
    r_lime = lime.explain(env, A)
    r_cose = cose.explain(env, A, guide_ranking=r_shap["ranking"])
    assert "ranking" in r_shap and len(r_shap["ranking"])>0
    assert "ranking" in r_lime and len(r_lime["ranking"])>0
    assert "cose_set" in r_cose

def test_eval_metrics():
    m = importlib.import_module("eval.metrics")
    from envs.generator import generate_environment
    from planners.a_star import AStarPlanner
    from explainers.baselines import geodesic_line_ranking
    env = generate_environment(H=20, W=20, density=0.35)
    A = AStarPlanner(connectivity=8)
    if A.plan(env.grid, env.start, env.goal)["success"]:
        # force failure by blocking the start ring
        env.grid[env.start[0]:env.start[0]+2, env.start[1]:env.start[1]+2] = True
    ranking = geodesic_line_ranking(env, A)["ranking"]
    out = m.evaluate_ranking_success_curve(env, A, ranking, k_max=3)
    assert "success_at_k" in out and "auc_s_at_k" in out

def test_cli_help():
    import subprocess, sys
    for mod in ["cli.run_eval","cli.run_exact_small","cli.run_transfer","cli.run_robustness","cli.make_figs"]:
        r = subprocess.run([sys.executable, "-m", mod, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert r.returncode == 0, f"{mod} --help failed"

if __name__ == "__main__":
    check("envs", test_envs)
    check("planners", test_planner)
    check("explainers", test_explainers)
    check("eval.metrics", test_eval_metrics)
    check("CLIs --help", test_cli_help)
    print(f"[{OK}] All self-checks passed.")
