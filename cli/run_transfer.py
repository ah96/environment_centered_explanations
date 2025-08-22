#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_transfer.py
---------------
Cross-planner transfer evaluation:
  (1) Apply explanations (rankings) computed under planner A to planner B
      and measure Success@k and AUC-S@K ("transfer success").
  (2) Compare explanations computed separately under A vs B via
      Jaccard(top-k) and Kendall's tau (ranking correlation).

Example:
  python -m explanations.cli.run_transfer \
      --sizes 20x20,30x30 --densities 0.10,0.20 \
      --num-envs 40 \
      --planners a_star,dijkstra,bfs,dfs,theta_star \
      --explainers shap,lime,rand,geodesic \
      --kmax 5 --seed 0
"""

from __future__ import annotations
import argparse, csv, os, time
from typing import Dict, List, Tuple, Optional
import numpy as np

# --- Flexible imports (package vs flat) ---
try:
    from explanations.envs.generator import generate_environment, GridEnvironment
    from explanations.planners.a_star import AStarPlanner
    from explanations.planners.dijkstra import DijkstraPlanner
    from explanations.planners.bfs import BFSPlanner
    from explanations.planners.dfs import DFSPlanner
    from explanations.planners.theta_star import ThetaStarPlanner
    from explanations.explainers.lime_explainer import LimeExplainer
    from explanations.explainers.shap_explainer import ShapExplainer
    from explanations.explainers.baselines import random_ranking, geodesic_line_ranking
    from explanations.eval.metrics import evaluate_ranking_success_curve, topk_set, jaccard, kendall_tau
    from explanations.eval.transfer import cross_planner_success_at_k, cross_planner_overlap
except Exception:
    from envs.generator import generate_environment, GridEnvironment  # type: ignore
    from planners.a_star import AStarPlanner  # type: ignore
    from planners.dijkstra import DijkstraPlanner  # type: ignore
    from planners.bfs import BFSPlanner  # type: ignore
    from planners.dfs import DFSPlanner  # type: ignore
    from planners.theta_star import ThetaStarPlanner  # type: ignore
    from explainers.lime_explainer import LimeExplainer  # type: ignore
    from explainers.shap_explainer import ShapExplainer  # type: ignore
    from explainers.baselines import random_ranking, geodesic_line_ranking  # type: ignore
    from eval.metrics import evaluate_ranking_success_curve, topk_set, jaccard, kendall_tau  # type: ignore
    from eval.transfer import cross_planner_success_at_k, cross_planner_overlap  # type: ignore


PLANNER_ALIASES = {
    "a_star": AStarPlanner,
    "dijkstra": DijkstraPlanner,
    "bfs": BFSPlanner,
    "dfs": DFSPlanner,
    "theta_star": ThetaStarPlanner,
}

def _parse_sizes(s: str) -> List[Tuple[int,int]]:
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        h, w = tok.split("x")
        out.append((int(h), int(w)))
    return out

def _parse_densities(s: str) -> List[float]:
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        vals.append(float(tok[:-1])/100.0 if tok.endswith("%") else float(tok))
    return vals

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _planner_to_bool(res) -> bool:
    if isinstance(res, dict): return bool(res.get("success", False))
    return bool(res)

def _generate_failure_env(H: int, W: int, density: float, max_tries: int = 80, rng=None):
    """
    Generate a random environment of size HxW at given density that actually FAILS
    under a simple reference planner (A* 8-connected). We DO NOT pass any 'seed'
    or 'ensure_*' kwargs because generator.py does not accept them.

    Returns:
        env (GridEnvironment) or None if we couldn't find a failing env in max_tries.
    """
    from planners.a_star import AStarPlanner
    a_star = AStarPlanner(connectivity=8)
    for _ in range(max_tries):
        try:
            env = generate_environment(H=H, W=W, density=density, rng=rng)
        except TypeError:
            env = generate_environment(H, W, density, rng=rng)
        ok = _planner_to_bool(a_star.plan(env.grid, env.start, env.goal))
        if not ok:
            return env
    return None


def _make_ranking(method: str, env, planner, lime_ns, lime_fp, shap_perm, seed, focus_top_m: int = 0) -> Dict:
    if method == "lime":
        expl = LimeExplainer(num_samples=lime_ns, flip_prob=lime_fp,
                             random_state=seed, focus_top_m=(focus_top_m or None))
        return expl.explain(env, planner)
    elif method == "shap":
        expl = ShapExplainer(permutations=shap_perm, random_state=seed,
                             focus_top_m=(focus_top_m or None))
        return expl.explain(env, planner)
    elif method == "rand":
        return random_ranking(env, random_state=seed)
    elif method == "geodesic":
        return geodesic_line_ranking(env)
    else:
        raise ValueError(f"Unknown explainer '{method}'")



def main():
    ap = argparse.ArgumentParser(description="Cross-planner transfer evaluation.")
    ap.add_argument("--sizes", default="20x20,30x30", type=str)
    ap.add_argument("--densities", default="0.10,0.20,0.30", type=str)
    ap.add_argument("--num-envs", default=40, type=int)
    ap.add_argument("--planners", default="a_star,dijkstra,bfs,dfs,theta_star", type=str,
                    help="Comma-separated planners to include.")
    ap.add_argument("--explainers", default="shap,lime,rand,geodesic", type=str,
                    help="Comma-separated ranking explainers (no COSE here).")
    ap.add_argument("--kmax", default=5, type=int)
    ap.add_argument("--connectivity", default=8, type=int, choices=[4,8])
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lime-samples", default=500, type=int)
    ap.add_argument("--lime-flip", default=0.30, type=float)
    ap.add_argument("--shap-perm", default=100, type=int)
    ap.add_argument("--outdir", default="results/csv", type=str)
    ap.add_argument("--focus-top-m", type=int, default=0, help="If >0, restrict SHAP/LIME to the top-M obstacles from a cheap geodesic heuristic.")
    args = ap.parse_args()

    import numpy as np, random
    np.random.seed(args.seed)   # makes generate_environment(...) deterministic
    random.seed(args.seed)      # if your code ever uses python's random

    sizes = _parse_sizes(args.sizes)
    densities = _parse_densities(args.densities)
    planner_keys = [p.strip().lower() for p in args.planners.split(",")]
    expl_keys = [e.strip().lower() for e in args.explainers.split(",")]

    # instantiate planners
    planners = {}
    for key in planner_keys:
        if key not in PLANNER_ALIASES:
            raise ValueError(f"Unknown planner {key}")
        planners[key] = PLANNER_ALIASES[key](connectivity=args.connectivity)

    _ensure_dir(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    import os as os_mod  # avoid any accidental local shadowing of `os`
    out_csv = os_mod.path.join(args.outdir, f"transfer_{stamp}.csv")

    fields = [
        "env_id","H","W","density",
        "planner_A","planner_B","method",
        "kmax",
        "transfer_success_at_k", "transfer_auc_s_at_k",   # A's ranking applied to B
        "overlap_topk", "kendall_tau",                   # A vs B explanations compared directly
        "calls_A","time_A","calls_B","time_B"
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        base_rng = np.random.default_rng(args.seed)
        env_id = 0

        for (H,W) in sizes:
            for dens in densities:
                for i in range(args.num_envs):
                    base = (int(args.seed) * 1_000_003 + int(env_id) * 97 + int(H) * 11 + int(W) * 13 + int(round(dens*1000)) * 17) % 2**32
                    rng = np.random.default_rng(base)
                    env = _generate_failure_env(H, W, dens, rng=rng)

                    if env is None:
                        continue
                    env_id += 1

                    # after generating env
                    import numpy as np
                    os_mod.makedirs("results/envs", exist_ok=True)
                    np.savez_compressed(f"results/envs/E_{H}x{W}_d{dens}_{env_id}.npz",
                                        grid=env.grid, obj_map=env.obj_map, start=env.start, goal=env.goal)

                    # precompute initial success for each planner to decide transfer success applicability
                    initial_ok = {k: _planner_to_bool(p.plan(env.grid, env.start, env.goal))
                                  for k,p in planners.items()}

                    for a_key, A in planners.items():
                        # skip if planner A already succeeds (still can compute ranking, but less meaningful)
                        # We'll still compute rankings and similarity; transfer success@k to B only if B fails.
                        for b_key, B in planners.items():
                            if a_key == b_key:  # skip diagonal
                                continue

                            for m in expl_keys:
                                seed_i = int(base_rng.integers(1e9))

                                # ranking under A
                                resA = _make_ranking(m, env, A, args.lime_samples, args.lime_flip, args.shap_perm, seed_i)
                                rankA = resA["ranking"]

                                # ranking under B (for similarity)
                                resB = _make_ranking(m, env, B, args.lime_samples, args.lime_flip, args.shap_perm, seed_i+1)
                                rankB = resB["ranking"]

                                # transfer success (only meaningful if B fails initially)
                                K = int(min(args.kmax, max(1, len(env.obstacles))))
                                if not initial_ok[b_key]:
                                    curveB = cross_planner_success_at_k(env, rankA, B, ks=list(range(1, K+1)))
                                    # simple normalized AUC
                                    aucB = float(np.mean([curveB[k] for k in sorted(curveB)])) if curveB else 0.0
                                else:
                                    curveB = {}
                                    aucB = ""

                                # similarity metrics (always computable)
                                jac = jaccard(topk_set(rankA, K), topk_set(rankB, K))
                                tau = kendall_tau(rankA, rankB)

                                row = {
                                    "env_id": env_id, "H": H, "W": W, "density": dens,
                                    "planner_A": a_key, "planner_B": b_key, "method": m,
                                    "kmax": K,
                                    "transfer_success_at_k": repr(curveB),
                                    "transfer_auc_s_at_k": aucB,
                                    "overlap_topk": float(jac),
                                    "kendall_tau": float(tau),
                                    "calls_A": int(resA.get("calls",0)),
                                    "time_A": float(resA.get("time_sec",0.0)),
                                    "calls_B": int(resB.get("calls",0)),
                                    "time_B": float(resB.get("time_sec",0.0)),
                                }
                                w.writerow(row)

    print(f"[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
