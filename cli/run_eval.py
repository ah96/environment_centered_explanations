#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval.py
-----------
Main evaluation loop:
- Generates random environments across (sizes × densities × seeds)
- Runs selected planners; filters to failure cases
- Computes explanations (LIME, SHAP, COSE, baselines)
- Evaluates Success@k and AUC-S@K (+ optional robustness metrics)
- Writes results to CSV in explanations/results/csv/

Example:
    python -m explanations.cli.run_eval \
        --sizes 20x20,30x30 \
        --densities 0.10,0.20 \
        --num-envs 50 \
        --planners a_star,dijkstra,bfs,dfs,theta_star \
        --explainers shap,lime,cose,rand,geodesic \
        --kmax 5 \
        --seed 0 \
        --robustness false

Grid convention: grid[r,c] == True means obstacle (blocked), False means free.
"""

from __future__ import annotations
import argparse
import csv
import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------- Flexible imports (package vs. flat folder) ---------- #
try:
    from envs.generator import generate_environment, GridEnvironment
    from explainers.lime_explainer import LimeExplainer
    from explainers.shap_explainer import ShapExplainer
    from explainers.cose import COSEExplainer
    from explainers.baselines import random_ranking, geodesic_line_ranking
    from planners.a_star import AStarPlanner
    from planners.dijkstra import DijkstraPlanner
    from planners.bfs import BFSPlanner
    from planners.dfs import DFSPlanner
    from planners.theta_star import ThetaStarPlanner
    from eval.metrics import evaluate_ranking_success_curve, topk_set, jaccard, kendall_tau, sum_calls, sum_time_sec
    #from eval.robustness import robustness_suite
except Exception:
    pass

# -------------------- helpers -------------------- #

PLANNER_ALIASES = {
    "a_star": AStarPlanner,
    "dijkstra": DijkstraPlanner,
    "bfs": BFSPlanner,
    "dfs": DFSPlanner,
    "theta_star": ThetaStarPlanner,
}

def _parse_sizes(s: str) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    for token in s.split(","):
        token = token.strip().lower()
        if "x" not in token:
            raise ValueError(f"Bad size '{token}', expected like 30x30")
        h, w = token.split("x")
        sizes.append((int(h), int(w)))
    return sizes

def _parse_densities(s: str) -> List[float]:
    vals = []
    for token in s.split(","):
        token = token.strip()
        if token.endswith("%"):
            vals.append(float(token[:-1]) / 100.0)
        else:
            vals.append(float(token))
    return vals

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _planner_to_bool(res) -> bool:
    if isinstance(res, dict):
        return bool(res.get("success", False))
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



# -------------------- main loop -------------------- #

def main():
    ap = argparse.ArgumentParser(description="Run environment-centered explanation evaluation.")
    ap.add_argument("--sizes", type=str, default="20x20,30x30",
                    help="Comma-separated grid sizes like 20x20,30x30")
    ap.add_argument("--densities", type=str, default="0.10,0.20,0.30",
                    help="Comma-separated densities (0–1 or %%, e.g., 10%%)")
    ap.add_argument("--num-envs", type=int, default=50, help="Environments per (size,density)")
    ap.add_argument("--planners", type=str, default="a_star,dijkstra,bfs,dfs,theta_star",
                    help="Comma-separated planners: a_star,dijkstra,bfs,dfs,theta_star")
    ap.add_argument("--explainers", type=str, default="shap,lime,cose,rand,geodesic",
                    help="Comma-separated explainers: shap,lime,cose,rand,geodesic")
    ap.add_argument("--kmax", type=int, default=5, help="Max k for Success@k curve")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    ap.add_argument("--connectivity", type=int, default=8, choices=[4,8], help="Grid connectivity for planners")
    ap.add_argument("--lime-samples", type=int, default=500, help="LIME #samples")
    ap.add_argument("--lime-flip", type=float, default=0.30, help="LIME bit-flip probability")
    ap.add_argument("--shap-perm", type=int, default=100, help="SHAP #permutations")
    ap.add_argument("--robustness", type=str, default="false",
                    help="Whether to compute robustness metrics (true/false)")
    ap.add_argument("--outdir", type=str, default="results/csv", help="Output directory for CSV")
    ap.add_argument("--focus-top-m", type=int, default=0, help="If >0, restrict SHAP/LIME to the top-M obstacles from a cheap geodesic heuristic.")
    args = ap.parse_args()

    import random
    np.random.seed(args.seed)   # makes generate_environment(...) deterministic
    random.seed(args.seed)      

    do_robust = str(args.robustness).strip().lower() in {"1","true","yes","y"}
    if do_robust:
        # lazy import; requires robustness module
        from eval.robustness import robustness_suite

    sizes = _parse_sizes(args.sizes)
    densities = _parse_densities(args.densities)
    planners_sel = [p.strip().lower() for p in args.planners.split(",")]
    explainers_sel = [e.strip().lower() for e in args.explainers.split(",")]

    # Instantiate planner objects once (reused across envs)
    planners = {}
    for key in planners_sel:
        if key not in PLANNER_ALIASES:
            raise ValueError(f"Unknown planner '{key}'")
        planners[key] = PLANNER_ALIASES[key](connectivity=args.connectivity)

    # Prepare output CSV (unique, atomic)
    _ensure_dir(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_tag = f"s{args.seed}"
    out_csv = os.path.join(args.outdir, f"eval_{safe_tag}_{stamp}.csv")
    tmp_csv = out_csv + f".tmp_{os.getpid()}"
    fieldnames = [
        "env_id","H","W","density","planner","method",
        "kmax","success_at_k","auc_s_at_k",
        "expl_set_size","calls","time_sec",
        "robust_jaccard","robust_kendall"
    ]

    with open(tmp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        env_id = 0
        base_rng = np.random.default_rng(args.seed)

        for (H,W) in sizes:
            for dens in densities:
                for i_env in range(args.num_envs):
                    # Generate unique seed for this environment
                    base = (int(args.seed) * 1_000_003 + int(env_id) * 97 + int(H) * 11 + int(W) * 13 + int(round(dens*1000)) * 17) % 2**32
                    rng = np.random.default_rng(base)
                    env = _generate_failure_env(H, W, dens, rng=rng)

                    if env is None:
                        # Could not find a failing env within tries; skip
                        continue
                    env_id += 1

                    # Save environment for reproducibility/debugging
                    os.makedirs("results/envs", exist_ok=True)
                    np.savez_compressed(f"results/envs/E_{H}x{W}_d{dens}_{env_id}.npz",
                                        grid=env.grid, obj_map=env.obj_map, start=env.start, goal=env.goal)

                    for pkey, planner in planners.items():
                        # Only evaluate failure cases for this planner; skip if already success
                        if _planner_to_bool(planner.plan(env.grid, env.start, env.goal)):
                            continue

                        # Prepare explainers (instantiate fresh to avoid RNG coupling)
                        results: Dict[str, Dict] = {}
                        rankings: Dict[str, List[Tuple[int,float]]] = {}

                        if "lime" in explainers_sel:
                            lime = LimeExplainer(num_samples=args.lime_samples,
                                                flip_prob=args.lime_flip,
                                                random_state=base,
                                                focus_top_m=(args.focus_top_m or None))
                            results["lime"] = lime.explain(env, planner)
                            rankings["lime"] = results["lime"]["ranking"]

                        if "shap" in explainers_sel:
                            shapx = ShapExplainer(permutations=args.shap_perm,
                                                random_state=base,
                                                focus_top_m=(args.focus_top_m or None))
                            results["shap"] = shapx.explain(env, planner)
                            rankings["shap"] = results["shap"]["ranking"]

                        if "rand" in explainers_sel:
                            results["rand"] = random_ranking(env, random_state=base)
                            rankings["rand"] = results["rand"]["ranking"]

                        if "geodesic" in explainers_sel:
                            results["geodesic"] = geodesic_line_ranking(env)
                            rankings["geodesic"] = results["geodesic"]["ranking"]

                        # COSE (can be guided by SHAP or LIME if available)
                        if "cose" in explainers_sel:
                            guide = rankings.get("shap") or rankings.get("lime") or None
                            cose = COSEExplainer(guide="shap" if "shap" in rankings else ("lime" if "lime" in rankings else "none"))
                            results["cose"] = cose.explain(env, planner, guide_ranking=guide)

                        # ----- Write per-method metrics -----
                        K = int(min(args.kmax, max(1, len(env.obstacles))))
                        ks = list(range(1, K+1))

                        # Ranking-based methods
                        for mkey in ["shap","lime","rand","geodesic"]:
                            if mkey not in results:
                                continue
                            curve = evaluate_ranking_success_curve(env, planner, rankings[mkey], k_max=K)
                            row = {
                                "env_id": env_id,
                                "H": H, "W": W, "density": dens,
                                "planner": pkey, "method": mkey,
                                "kmax": K,
                                "success_at_k": repr(curve["success_at_k"]),
                                "auc_s_at_k": float(curve["auc_s_at_k"]),
                                "expl_set_size": "",  # N/A for pure rankings
                                "calls": int(results[mkey].get("calls", 0)),
                                "time_sec": float(results[mkey].get("time_sec", 0.0)),
                                "robust_jaccard": "",
                                "robust_kendall": "",
                            }

                            # Optional robustness: compare original ranking to perturbed envs
                            if do_robust:
                                try:
                                    r0 = rankings[mkey]
                                    jac_list = []
                                    tau_list = []
                                    for kind, envp in robustness_suite(env, seed=base):
                                        # recompute ranking under perturbed env
                                        if mkey == "shap":
                                            rx = ShapExplainer(permutations=max(30, args.shap_perm//2), random_state=base+1).explain(envp, planner)["ranking"]
                                        elif mkey == "lime":
                                            rx = LimeExplainer(num_samples=max(200, args.lime_samples//2), flip_prob=args.lime_flip, random_state=base+1).explain(envp, planner)["ranking"]
                                        elif mkey == "rand":
                                            rx = random_ranking(envp, random_state=base+1)["ranking"]
                                        else:
                                            rx = geodesic_line_ranking(envp)["ranking"]
                                        # compute stability
                                        tau = kendall_tau(r0, rx)
                                        # Jaccard on top-K (same K)
                                        jac = jaccard(topk_set(r0, K), topk_set(rx, K))
                                        tau_list.append(tau); jac_list.append(jac)
                                    row["robust_jaccard"] = float(np.mean(jac_list)) if jac_list else ""
                                    row["robust_kendall"] = float(np.mean(tau_list)) if tau_list else ""
                                except Exception:
                                    pass

                            writer.writerow(row)

                        # COSE (counterfactual set)
                        if "cose" in results:
                            cose_out = results["cose"]
                            cose_set = set(cose_out.get("cose_set", []))
                            row = {
                                "env_id": env_id,
                                "H": H, "W": W, "density": dens,
                                "planner": pkey, "method": "cose",
                                "kmax": "", "success_at_k": "",
                                "auc_s_at_k": "",
                                "expl_set_size": len(cose_set),
                                "calls": int(cose_out.get("calls", 0)),
                                "time_sec": float(cose_out.get("time_sec", 0.0)),
                                "robust_jaccard": "",
                                "robust_kendall": "",
                            }

                            if do_robust:
                                try:
                                    jac_list = []
                                    for kind, envp in robustness_suite(env, seed=base):
                                        # Re-run COSE (guided by same guide style if available)
                                        guide = rankings.get("shap") or rankings.get("lime") or None
                                        cose_p = COSEExplainer(guide="shap" if "shap" in rankings else ("lime" if "lime" in rankings else "none"))
                                        outp = cose_p.explain(envp, planner, guide_ranking=guide)
                                        perturbed_set = set(outp.get("cose_set", []))
                                        jac_list.append(jaccard(cose_set, perturbed_set))
                                    row["robust_jaccard"] = float(np.mean(jac_list)) if jac_list else ""
                                except Exception:
                                    pass

                            writer.writerow(row)

    # Atomic rename to final path
    os.replace(tmp_csv, out_csv)
    print(f"[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()