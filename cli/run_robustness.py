#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_robustness.py
-----------------
Robustness evaluation under environment perturbations:
  - jitter (small obstacle translations)
  - dilate / erode (morphology)
  - distractors (add small far-away obstacles)

Metrics:
  - Ranking methods (LIME/SHAP/Rand/Geodesic): Kendall's tau (ranking stability),
    Jaccard(top-k) overlap.
  - COSE (sets): Jaccard(set) overlap.

Example:
  python -m cli.run_robustness \
      --sizes 30x30 --densities 0.20 --num-envs 50 \
      --planners a_star,dijkstra \
      --explainers shap,lime,cose,rand,geodesic \
      --kmax 5 --seed 0
"""

import argparse, csv, os, time
import random  # Add this import at the top
from typing import Dict, List, Tuple
import numpy as np

# --- Flexible imports ---
try:
    from envs.generator import generate_environment, GridEnvironment
    from planners.a_star import AStarPlanner
    from planners.dijkstra import DijkstraPlanner
    from planners.bfs import BFSPlanner
    from planners.dfs import DFSPlanner
    from planners.theta_star import ThetaStarPlanner
    from explainers.lime_explainer import LimeExplainer
    from explainers.shap_explainer import ShapExplainer
    from explainers.cose import COSEExplainer
    from explainers.baselines import random_ranking, geodesic_line_ranking
    from eval.metrics import topk_set, jaccard, kendall_tau, _planner_to_bool
    from eval.robustness import robustness_suite
except Exception:
    # Package-style fallbacks
    from explanations.envs.generator import generate_environment, GridEnvironment  # type: ignore
    from explanations.planners.a_star import AStarPlanner  # type: ignore
    from explanations.planners.dijkstra import DijkstraPlanner  # type: ignore
    from explanations.planners.bfs import BFSPlanner  # type: ignore
    from explanations.planners.dfs import DFSPlanner  # type: ignore
    from explanations.planners.theta_star import ThetaStarPlanner  # type: ignore
    from explanations.explainers.lime_explainer import LimeExplainer  # type: ignore
    from explanations.explainers.shap_explainer import ShapExplainer  # type: ignore
    from explanations.explainers.cose import COSEExplainer  # type: ignore
    from explanations.explainers.baselines import random_ranking, geodesic_line_ranking  # type: ignore
    from explanations.eval.metrics import topk_set, jaccard, kendall_tau, _planner_to_bool  # type: ignore
    from explanations.eval.robustness import robustness_suite  # type: ignore


PLANNER_ALIASES = {
    "a_star": AStarPlanner,
    "dijkstra": DijkstraPlanner,
    "bfs": BFSPlanner,
    "dfs": DFSPlanner,
    "theta_star": ThetaStarPlanner,
}

def _match_ids_by_iou(envA, envB, iou_thresh=0.3):
    idsA = [i for i in np.unique(envA.obj_map) if i>0]
    idsB = [i for i in np.unique(envB.obj_map) if i>0]
    M = np.zeros((len(idsA), len(idsB)), float)
    for a_i, a in enumerate(idsA):
        A = (envA.obj_map == a)
        a_area = int(A.sum())
        if a_area == 0: continue
        for b_j, b in enumerate(idsB):
            B = (envB.obj_map == b)
            inter = int((A & B).sum())
            union = a_area + int(B.sum()) - inter
            M[a_i, b_j] = (inter/union) if union>0 else 0.0
    mapping = {}
    usedB = set()
    for a_i, a in enumerate(idsA):
        j = M[a_i].argmax() if M.shape[1]>0 else -1
        b = idsB[j] if j>=0 else None
        if j>=0 and M[a_i, j] >= iou_thresh and b not in usedB:
            mapping[a] = b; usedB.add(b)
        else:
            mapping[a] = None
    return mapping

def _remap_ranking(ranking, id_map):
    from collections import defaultdict
    acc = defaultdict(float)
    for oid, s in ranking:
        b = id_map.get(int(oid))
        if b is not None:
            acc[int(b)] += float(s)
    return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))

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

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _generate_failure_env(H: int, W: int, density: float, max_tries: int = 80, rng=None):
    """
    Generate a random HxW environment at given density that FAILS under A* (8-connected).
    No seed or ensure_* kwargs — we just sample until failure (or give up).
    Returns: GridEnvironment or None if not found within max_tries.
    """
    try:
        # local (flat) imports
        from planners.a_star import AStarPlanner as _AStar
        from eval.metrics import _planner_to_bool as _ptb
        from envs.generator import generate_environment as _gen
    except Exception:
        # package-style fallbacks
        from explanations.planners.a_star import AStarPlanner as _AStar  # type: ignore
        from explanations.eval.metrics import _planner_to_bool as _ptb  # type: ignore
        from explanations.envs.generator import generate_environment as _gen  # type: ignore

    a_star = _AStar(connectivity=8)
    for _ in range(max_tries):
        try:
            env = _gen(H=H, W=W, density=density, rng=rng)
        except TypeError:
            env = _gen(H, W, density, rng=rng)
        if not _ptb(a_star.plan(env.grid, env.start, env.goal)):
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
    ap = argparse.ArgumentParser(description="Robustness evaluation under environment perturbations.")
    ap.add_argument("--sizes", default="20x20,30x30", type=str)
    ap.add_argument("--densities", default="0.10,0.20,0.30", type=str)
    ap.add_argument("--num-envs", default=40, type=int)
    ap.add_argument("--planners", default="a_star,dijkstra,bfs,dfs,theta_star", type=str)
    ap.add_argument("--explainers", default="shap,lime,cose,rand,geodesic", type=str)
    ap.add_argument("--kmax", default=5, type=int)
    ap.add_argument("--connectivity", default=8, type=int, choices=[4,8])
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lime-samples", default=500, type=int)
    ap.add_argument("--lime-flip", default=0.30, type=float)
    ap.add_argument("--shap-perm", default=100, type=int)
    ap.add_argument("--outdir", default="results/csv", type=str)
    ap.add_argument("--geom-remap", action="store_true", help="Match obstacle IDs across planners by IoU before comparing rankings.")
    ap.add_argument("--focus-top-m", type=int, default=0, help="If >0, restrict SHAP/LIME to the top-M obstacles from a cheap geodesic heuristic.")
    args = ap.parse_args()

    import random
    np.random.seed(args.seed)
    random.seed(args.seed)

    sizes = _parse_sizes(args.sizes)
    densities = _parse_densities(args.densities)
    planner_keys = [p.strip().lower() for p in args.planners.split(",")]
    expl_keys = [e.strip().lower() for e in args.explainers.split(",")]

    planners = {}
    for key in planner_keys:
        if key not in PLANNER_ALIASES:
            raise ValueError(f"Unknown planner {key}")
        planners[key] = PLANNER_ALIASES[key](connectivity=args.connectivity)

    _ensure_dir(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    import os as os_mod  # avoid shadowing
    safe_tag = f"s{args.seed}"
    out_csv = os_mod.path.join(args.outdir, f"robustness_{safe_tag}_{stamp}.csv")
    tmp_csv = out_csv + f".tmp_{os_mod.getpid()}"

    fields = [
        "env_id","H","W","density","planner","method","perturbation",
        "kmax",
        "kendall_tau",             # for ranking methods
        "jaccard_topk",            # for ranking methods
        "cose_jaccard",            # for COSE
        "calls_base","time_base","calls_pert","time_pert"
    ]

    with open(tmp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        base_rng = np.random.default_rng(args.seed)
        env_id = 0

        for (H,W) in sizes:
            for dens in densities:
                for _ in range(args.num_envs):
                    base = (int(args.seed) * 1_000_003 + int(env_id) * 97 + int(H) * 11 + int(W) * 13 + int(round(dens*1000)) * 17) % 2**32
                    rng = np.random.default_rng(base)
                    env = _generate_failure_env(H, W, dens, rng=rng)
                    if env is None:
                        continue
                    env_id += 1

                    # snapshot (useful for debugging/repro)
                    os_mod.makedirs("results/envs", exist_ok=True)
                    np.savez_compressed(f"results/envs/E_{H}x{W}_d{dens}_{env_id}.npz",
                                        grid=env.grid, obj_map=env.obj_map, start=env.start, goal=env.goal)

                    for pkey, planner in planners.items():
                        # Only evaluate failure cases for this planner
                        if _planner_to_bool(planner.plan(env.grid, env.start, env.goal)):
                            continue

                        # Base results once per method
                        base_rankings: Dict[str, List[Tuple[int,float]]] = {}
                        base_records: Dict[str, Dict] = {}

                        if "shap" in expl_keys:
                            res = ShapExplainer(permutations=args.shap_perm,
                                                random_state=int(base_rng.integers(1e9)),
                                                focus_top_m=(args.focus_top_m or None)).explain(env, planner)
                            base_rankings["shap"] = res["ranking"]; base_records["shap"] = res

                        if "lime" in expl_keys:
                            res = LimeExplainer(num_samples=args.lime_samples, flip_prob=args.lime_flip,
                                                random_state=int(base_rng.integers(1e9)),
                                                focus_top_m=(args.focus_top_m or None)).explain(env, planner)
                            base_rankings["lime"] = res["ranking"]; base_records["lime"] = res

                        if "rand" in expl_keys:
                            res = random_ranking(env, random_state=int(base_rng.integers(1e9)))
                            base_rankings["rand"] = res["ranking"]; base_records["rand"] = res

                        if "geodesic" in expl_keys:
                            res = geodesic_line_ranking(env)
                            base_rankings["geodesic"] = res["ranking"]; base_records["geodesic"] = res

                        if "cose" in expl_keys:
                            guide = base_rankings.get("shap") or base_rankings.get("lime") or None
                            cose = COSEExplainer(guide="shap" if "shap" in base_rankings else ("lime" if "lime" in base_rankings else "none"))
                            res = cose.explain(env, planner, guide_ranking=guide)
                            base_records["cose"] = res

                        variants = robustness_suite(env, seed=int(base_rng.integers(1e9)))
                        K = int(min(args.kmax, max(1, len(env.obstacles))))

                        for kind, envp in variants:
                            # Ranking methods
                            for m in ["shap","lime","rand","geodesic"]:
                                if m not in base_rankings or m not in expl_keys:
                                    continue
                                seed_i = int(base_rng.integers(1e9))
                                resp = _make_ranking(
                                    m, envp, planner,
                                    args.lime_samples//2 if m=="lime" else args.lime_samples,
                                    args.lime_flip,
                                    args.shap_perm//2 if m=="shap" else args.shap_perm,
                                    seed_i,
                                    focus_top_m=args.focus_top_m
                                )
                                rank0 = base_rankings[m]; rankp = resp["ranking"]
                                tau = kendall_tau(rank0, rankp)
                                jac = jaccard(topk_set(rank0, K), topk_set(rankp, K))
                                w.writerow({
                                    "env_id": env_id, "H": H, "W": W, "density": dens,
                                    "planner": pkey, "method": m, "perturbation": kind,
                                    "kmax": K,
                                    "kendall_tau": float(tau),
                                    "jaccard_topk": float(jac),
                                    "cose_jaccard": "",
                                    "calls_base": int(base_records[m].get("calls",0)),
                                    "time_base": float(base_records[m].get("time_sec",0.0)),
                                    "calls_pert": int(resp.get("calls",0)),
                                    "time_pert": float(resp.get("time_sec",0.0)),
                                })

                            # COSE: set Jaccard
                            if "cose" in expl_keys and "cose" in base_records:
                                guide = base_rankings.get("shap") or base_rankings.get("lime") or None
                                cosep = COSEExplainer(guide="shap" if "shap" in base_rankings else ("lime" if "lime" in base_rankings else "none"))
                                outp = cosep.explain(envp, planner, guide_ranking=guide)
                                set0 = set(base_records["cose"].get("cose_set", set()))
                                setp = set(outp.get("cose_set", set()))
                                w.writerow({
                                    "env_id": env_id, "H": H, "W": W, "density": dens,
                                    "planner": pkey, "method": "cose", "perturbation": kind,
                                    "kmax": "",
                                    "kendall_tau": "",
                                    "jaccard_topk": "",
                                    "cose_jaccard": jaccard(set0, setp),
                                    "calls_base": int(base_records["cose"].get("calls",0)),
                                    "time_base": float(base_records["cose"].get("time_sec",0.0)),
                                    "calls_pert": int(outp.get("calls",0)),
                                    "time_pert": float(outp.get("time_sec",0.0)),
                                })

    # Atomic rename → final
    os_mod.replace(tmp_csv, out_csv)
    print(f"[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
