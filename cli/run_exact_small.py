#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_exact_small.py
------------------
Oracle minimality sweep on *small* grids (e.g., 20x20).
- Generates failing environments
- Computes ILP exact minimal obstacle removal set
- Runs COSE (guided by SHAP by default) and compares sizes
- Saves rows to CSV for optimality-gap analysis

Example:
  python -m cli.run_exact_small \
      --size 20x20 --density 0.20 --num-envs 100 \
      --planner a_star --connectivity 8 \
      --time-limit 60 --shap-perm 80 \
      --seed 0 --outdir results/csv
"""

from __future__ import annotations
import argparse, csv, os, time
from typing import Tuple
import numpy as np

# Prefer local (flat) imports; fall back to package-style if needed
try:
    from envs.generator import generate_environment, GridEnvironment  # type: ignore
    from planners.a_star import AStarPlanner  # type: ignore
    from planners.dijkstra import DijkstraPlanner  # type: ignore
    from planners.bfs import BFSPlanner  # type: ignore
    from planners.dfs import DFSPlanner  # type: ignore
    from planners.theta_star import ThetaStarPlanner  # type: ignore
    from explainers.shap_explainer import ShapExplainer  # type: ignore
    from explainers.cose import COSEExplainer  # type: ignore
    from eval.ilp_minimality import exact_minimal_set  # type: ignore
except Exception:
    from explanations.envs.generator import generate_environment, GridEnvironment  # type: ignore
    from explanations.planners.a_star import AStarPlanner  # type: ignore
    from explanations.planners.dijkstra import DijkstraPlanner  # type: ignore
    from explanations.planners.bfs import BFSPlanner  # type: ignore
    from explanations.planners.dfs import DFSPlanner  # type: ignore
    from explanations.planners.theta_star import ThetaStarPlanner  # type: ignore
    from explanations.explainers.shap_explainer import ShapExplainer  # type: ignore
    from explanations.explainers.cose import COSEExplainer  # type: ignore
    from explanations.eval.ilp_minimality import exact_minimal_set  # type: ignore

PLANNER_ALIASES = {
    "a_star": AStarPlanner,
    "dijkstra": DijkstraPlanner,
    "bfs": BFSPlanner,
    "dfs": DFSPlanner,
    "theta_star": ThetaStarPlanner,
}

def _parse_size(s: str) -> Tuple[int, int]:
    token = s.strip().lower()
    h, w = token.split("x")
    return int(h), int(w)

def _parse_density(s: str) -> float:
    s = s.strip()
    return float(s[:-1]) / 100.0 if s.endswith("%") else float(s)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _planner_to_bool(res) -> bool:
    if isinstance(res, dict):
        return bool(res.get("success", False))
    return bool(res)

def _generate_failure_env(H: int, W: int, density: float, max_tries: int = 80, rng=None):
    """
    Generate a random environment of size HxW at given density that FAILS
    under A* (8-connected). No seed/ensure_* kwargs (generator doesn't accept them).
    Returns: GridEnvironment or None if we couldn't find a failing env in max_tries.
    """
    try:
        from planners.a_star import AStarPlanner as _AStar
    except Exception:
        from explanations.planners.a_star import AStarPlanner as _AStar  # type: ignore
    a_star = _AStar(connectivity=8)
    for _ in range(max_tries):
        try:
            env = generate_environment(H=H, W=W, density=density, rng=rng)
        except TypeError:
            env = generate_environment(H, W, density, rng=rng)
        if not _planner_to_bool(a_star.plan(env.grid, env.start, env.goal)):
            return env
    return None

def main():
    ap = argparse.ArgumentParser(description="Exact minimality vs COSE on small maps (ILP oracle).")
    ap.add_argument("--size", type=str, default="20x20", help="Grid size HxW (e.g., 20x20)")
    ap.add_argument("--density", type=str, default="0.20", help="Obstacle density (0–1 or %%, e.g., 20%%)")
    ap.add_argument("--num-envs", type=int, default=50, help="Number of environments")
    ap.add_argument("--planner", type=str, default="a_star", help="Planner: a_star,dijkstra,bfs,dfs,theta_star")
    ap.add_argument("--connectivity", type=int, default=8, choices=[4,8], help="Planner connectivity")
    ap.add_argument("--time-limit", type=int, default=60, help="ILP time limit (seconds)")
    ap.add_argument("--shap-perm", type=int, default=80, help="SHAP permutations (guide for COSE)")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    ap.add_argument("--outdir", type=str, default="results/csv", help="Output directory")
    ap.add_argument("--focus-top-m", type=int, default=0,
                    help="If >0, restrict SHAP to the top-M obstacles from a geodesic heuristic.")
    args = ap.parse_args()

    import random
    np.random.seed(args.seed)   # makes generate_environment(...) deterministic
    random.seed(args.seed)      # if any plain-random code is used

    H, W = _parse_size(args.size)
    density = _parse_density(args.density)

    planner_cls = PLANNER_ALIASES.get(args.planner.lower())
    if planner_cls is None:
        raise ValueError(f"Unknown planner '{args.planner}'")
    planner = planner_cls(connectivity=args.connectivity)

    _ensure_dir(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    import os as os_mod  # avoid shadowing
    safe_tag = f"s{args.seed}"
    out_csv = os_mod.path.join(args.outdir, f"exact_small_{H}x{W}_{int(100*density)}_{safe_tag}_{stamp}.csv")
    tmp_csv = out_csv + f".tmp_{os_mod.getpid()}"

    fields = [
        "env_id","H","W","density","planner","connectivity",
        "ilp_status","ilp_obj","ilp_time_sec",
        "cose_size","cose_time_sec","gap","shap_calls","cose_calls"
    ]

    with open(tmp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        base_rng = np.random.default_rng(args.seed)
        env_id = 0
        for _ in range(args.num_envs):
            # deterministic per-env RNG derived from seed + env geometry + density
            base = (int(args.seed) * 1_000_003 + int(env_id) * 97 + int(H) * 11
                    + int(W) * 13 + int(round(density*1000)) * 17) % 2**32
            rng = np.random.default_rng(base)

            env = _generate_failure_env(H, W, density, rng=rng)
            if env is None:
                continue
            env_id += 1

            # Save environment snapshot for repro
            os_mod.makedirs("results/envs", exist_ok=True)
            np.savez_compressed(f"results/envs/E_{H}x{W}_d{density}_{env_id}.npz",
                                grid=env.grid, obj_map=env.obj_map, start=env.start, goal=env.goal)

            # ILP oracle
            t0 = time.perf_counter()
            try:
                ilp = exact_minimal_set(env, connectivity=args.connectivity, time_limit=args.time_limit)
            except Exception as e:
                ilp = {'set': set(), 'status': f"Error:{type(e).__name__}",
                       'obj': float("nan"), 'time_sec': time.perf_counter() - t0}
            ilp_size = len(ilp.get("set", set()))

            # COSE guided by SHAP
            shapx = ShapExplainer(permutations=args.shap_perm,
                                  random_state=int(base_rng.integers(1e9)),
                                  focus_top_m=(args.focus_top_m or None))
            shap_out = shapx.explain(env, planner)

            cose = COSEExplainer(guide="shap")
            cose_out = cose.explain(env, planner, guide_ranking=shap_out["ranking"])
            cose_size = len(cose_out.get("cose_set", set()))

            w.writerow({
                "env_id": env_id,
                "H": H, "W": W, "density": density,
                "planner": args.planner, "connectivity": args.connectivity,
                "ilp_status": ilp.get("status",""),
                "ilp_obj": ilp.get("obj",""),
                "ilp_time_sec": ilp.get("time_sec",""),
                "cose_size": cose_size,
                "cose_time_sec": cose_out.get("time_sec",""),
                "gap": (cose_size - ilp_size) if isinstance(ilp_size, int) else "",
                "shap_calls": shap_out.get("calls",0),
                "cose_calls": cose_out.get("calls",0),
            })

    # Atomic rename to final path
    os_mod.replace(tmp_csv, out_csv)
    print(f"[OK] Wrote: {out_csv}")

if __name__ == "__main__":
    main()
