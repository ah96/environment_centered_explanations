#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_agreement_and_qual.py
--------------------------
Generates:
  1) q1_agreement_heatmap.pdf  — method-vs-method agreement (mean Jaccard on top-k)
  2) qual_case_original.pdf    — the chosen environment (grid)
  3) qual_case_lime.pdf        — LIME top-k highlighted
  4) qual_case_shap.pdf        — SHAP top-k highlighted
  5) qual_case_cose.pdf        — COSE set highlighted

We recompute explanations from saved env snapshots in results/envs/E_*.npz to avoid
changing the eval CSV schema.

Usage (example):
  python -m cli.plot_agreement_and_qual \
    --env-glob "results/envs/E_*.npz" \
    --outdir "runs/20250825_200002/figs" \
    --planner a_star --connectivity 8 \
    --k 5 --max-envs 200 \
    --lime-samples 500 --lime-flip 0.30 \
    --shap-perm 100 \
    --seed 0 \
    --qual-env "20x20:0.2:17"

Notes:
- --qual-env lets you force a specific environment by key "HxW:density:env_id"
  matching the filename suffix saved by your CLIs (see results/envs/E_... files).
- If --qual-env is omitted, we pick a deterministic "best" env (first in sorted list).
"""

from __future__ import annotations
import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------- Flexible imports (flat repo vs package name) --------- #
try:
    from envs.generator import GridEnvironment
    from planners.a_star import AStarPlanner
    from planners.dijkstra import DijkstraPlanner
    from planners.bfs import BFSPlanner
    from planners.dfs import DFSPlanner
    from planners.theta_star import ThetaStarPlanner
    from explainers.lime_explainer import LimeExplainer
    from explainers.shap_explainer import ShapExplainer
    from explainers.cose import COSEExplainer
    from explainers.baselines import geodesic_line_ranking, random_ranking
    from eval.metrics import topk_set, jaccard, kendall_tau
except Exception:
    from explanations.envs.generator import GridEnvironment  # type: ignore
    from explanations.planners.a_star import AStarPlanner  # type: ignore
    from explanations.planners.dijkstra import DijkstraPlanner  # type: ignore
    from explanations.planners.bfs import BFSPlanner  # type: ignore
    from explanations.planners.dfs import DFSPlanner  # type: ignore
    from explanations.planners.theta_star import ThetaStarPlanner  # type: ignore
    from explanations.explainers.lime_explainer import LimeExplainer  # type: ignore
    from explanations.explainers.shap_explainer import ShapExplainer  # type: ignore
    from explanations.explainers.cose import COSEExplainer  # type: ignore
    from explanations.explainers.baselines import geodesic_line_ranking, random_ranking  # type: ignore
    from explanations.eval.metrics import topk_set, jaccard, kendall_tau  # type: ignore


PLANNERS = {
    "a_star": AStarPlanner,
    "dijkstra": DijkstraPlanner,
    "bfs": BFSPlanner,
    "dfs": DFSPlanner,
    "theta_star": ThetaStarPlanner,
}


def _load_env(npz_path: str) -> GridEnvironment:
    Z = np.load(npz_path, allow_pickle=True)
    class _Env:
        pass
    env = _Env()
    env.grid = Z["grid"]
    env.obj_map = Z["obj_map"].item() if isinstance(Z["obj_map"], np.ndarray) else Z["obj_map"]
    env.start = tuple(Z["start"])
    env.goal = tuple(Z["goal"])
    # convenience
    env.H, env.W = env.grid.shape
    # reconstruct obstacle indices as ints (row-major ids)
    obs = []
    for (r, c), is_block in np.ndenumerate(env.grid):
        if is_block:
            obs.append(r * env.W + c)
    env.obstacles = obs
    return env  # type: ignore


def _parse_env_key_from_path(path: str) -> str:
    # Paths are like: results/envs/E_20x20_d0.2_17.npz  (density may be "0.2" or "0.20")
    b = os.path.basename(path)
    m = re.match(r"E_(\d+)x(\d+)_d([0-9.]+)_([0-9]+)\.npz", b)
    if not m:
        return b
    H, W, d, eid = m.groups()
    # normalize density to 1 decimal if possible
    try:
        d = str(float(d))
    except Exception:
        pass
    return f"{H}x{W}:{d}:{eid}"


def _ranking_topk(env, planner, method: str, k: int, lime_ns: int, lime_fp: float, shap_perm: int, seed: int):
    if method == "geodesic":
        r = geodesic_line_ranking(env)["ranking"]
    elif method == "rand":
        r = random_ranking(env, random_state=seed)["ranking"]
    elif method == "lime":
        expl = LimeExplainer(num_samples=lime_ns, flip_prob=lime_fp, random_state=seed)
        r = expl.explain(env, planner)["ranking"]
    elif method == "shap":
        expl = ShapExplainer(permutations=shap_perm, random_state=seed)
        r = expl.explain(env, planner)["ranking"]
    else:
        raise ValueError(f"unknown method {method}")
    return topk_set(r, k)


def _draw_grid(env, ax, title: str = "", highlight: set | None = None):
    H, W = env.H, env.W
    ax.imshow(~env.grid, cmap="gray", interpolation="none")  # free=True=white, obstacles black
    # start/goal
    sr, sc = env.start; gr, gc = env.goal
    ax.scatter([sc], [sr], marker="o", s=60, edgecolors="k", facecolors="none", linewidths=2, label="Start")
    ax.scatter([gc], [gr], marker="*", s=80, edgecolors="k", facecolors="yellow", linewidths=1.5, label="Goal")
    # highlight set as colored overlay
    if highlight:
        mask = np.zeros_like(env.grid, dtype=float)
        for idx in highlight:
            r, c = divmod(idx, W)
            mask[r, c] = 1.0
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.85, interpolation="none")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-glob", type=str, default="results/envs/E_*.npz")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--planner", type=str, default="a_star", choices=list(PLANNERS.keys()))
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--k", type=int, default=5, help="Top-k for ranking methods & overlays")
    ap.add_argument("--max-envs", type=int, default=200, help="Cap number of environments to average for agreement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lime-samples", type=int, default=500)
    ap.add_argument("--lime-flip", type=float, default=0.30)
    ap.add_argument("--shap-perm", type=int, default=100)
    ap.add_argument("--qual-env", type=str, default="", help='Optional explicit key "HxW:density:env_id"')
    args = ap.parse_args()

    np.random.seed(args.seed)

    _ensure_dir(args.outdir)

    # Planner
    planner = PLANNERS[args.planner](connectivity=args.connectivity)

    # Collect envs
    paths = sorted(glob.glob(args.env_glob))
    if not paths:
        raise SystemExit(f"No env files for pattern: {args.env_glob}")
    if args.max_envs > 0:
        paths = paths[: args.max_envs]

    # Agreement across methods
    methods = ["shap", "lime", "geodesic", "rand"]
    M = pd.DataFrame(0.0, index=methods, columns=methods)
    C = pd.DataFrame(0, index=methods, columns=methods)

    for p in paths:
        env = _load_env(p)

        # Skip environments that are trivially solvable by the planner (we want failures)
        ok = planner.plan(env.grid, env.start, env.goal)
        ok = bool(ok["success"] if isinstance(ok, dict) else ok)
        if ok:
            continue

        sets = {}
        for m in methods:
            try:
                s = _ranking_topk(env, planner, m, args.k, args.lime_samples, args.lime_flip, args.shap_perm, args.seed+1)
                sets[m] = set(s)
            except Exception:
                continue

        # accumulate Jaccard for each pair
        for a in methods:
            for b in methods:
                if a == b or a not in sets or b not in sets:
                    continue
                jac = jaccard(sets[a], sets[b])
                M.loc[a, b] += jac
                C.loc[a, b] += 1

    # average
    for a in methods:
        for b in methods:
            if a == b or C.loc[a, b] == 0:
                continue
            M.loc[a, b] /= max(1, C.loc[a, b])

    # plot heatmap
    fig = plt.figure(figsize=(5.6, 4.8))
    im = plt.imshow(M.values, origin="upper", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Mean Jaccard (top-k)")
    plt.xticks(np.arange(len(methods)), [m.upper() for m in methods], rotation=0)
    plt.yticks(np.arange(len(methods)), [m.upper() for m in methods])
    # annotate
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j: continue
            v = M.values[i, j]
            plt.text(j, i, f"{100*v:.0f}%", ha="center", va="center", fontsize=9)
    plt.title(f"Q1 Agreement Heatmap (planner={args.planner}, k={args.k})")
    plt.tight_layout()
    out_heat = os.path.join(args.outdir, "q1_agreement_heatmap.pdf")
    plt.savefig(out_heat, bbox_inches="tight"); plt.close()
    print(f"[q1_agreement] Wrote {out_heat}")

    # --------- Qualitative case study (four PDFs) --------- #
    # Pick env
    qual_path = None
    if args.qual_env:
        # find matching key
        for p in sorted(glob.glob(args.env_glob)):
            if _parse_env_key_from_path(p) == args.qual_env:
                qual_path = p; break
        if qual_path is None:
            raise SystemExit(f"Could not find --qual-env '{args.qual_env}' in {args.env_glob}")
    else:
        qual_path = sorted(glob.glob(args.env_glob))[0]
    env = _load_env(qual_path)

    # Base original
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _draw_grid(env, ax, title="Original (failure case)")
    out = os.path.join(args.outdir, "qual_case_original.pdf")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[qual] Wrote {out}")

    # SHAP
    shap = ShapExplainer(permutations=args.shap_perm, random_state=args.seed+11)
    shap_r = shap.explain(env, planner)["ranking"]
    shap_top = topk_set(shap_r, args.k)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _draw_grid(env, ax, title=f"SHAP (top-{args.k})", highlight=set(shap_top))
    out = os.path.join(args.outdir, "qual_case_shap.pdf")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[qual] Wrote {out}")

    # LIME
    lime = LimeExplainer(num_samples=args.lime_samples, flip_prob=args.lime_flip, random_state=args.seed+13)
    lime_r = lime.explain(env, planner)["ranking"]
    lime_top = topk_set(lime_r, args.k)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _draw_grid(env, ax, title=f"LIME (top-{args.k})", highlight=set(lime_top))
    out = os.path.join(args.outdir, "qual_case_lime.pdf")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[qual] Wrote {out}")

    # COSE (guided by SHAP)
    cose = COSEExplainer(guide="shap")
    cose_out = cose.explain(env, planner, guide_ranking=shap_r)
    cose_set = set(cose_out.get("cose_set", []))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _draw_grid(env, ax, title="COSE (counterfactual set)", highlight=cose_set)
    out = os.path.join(args.outdir, "qual_case_cose.pdf")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[qual] Wrote {out}")


if __name__ == "__main__":
    main()
