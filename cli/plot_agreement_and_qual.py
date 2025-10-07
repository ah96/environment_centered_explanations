#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_agreement_and_qual.py
==========================

Server-optimized script to compute *agreement* between explanation methods 
(SHAP, LIME, geodesic baseline, random) on robot path planning failures,
and optionally generate qualitative case study plots.

Features:
---------
- Multiprocessing: runs in parallel using --workers (default: 8).
- Progress logging: prints progress/ETA every few files and heartbeat every N seconds.
- Safe for servers: uses headless matplotlib backend (Agg).
- Optional speed-ups: 
    * --max-envs      → cap total environments scanned
    * --max-failures  → stop after N failure cases
    * --skip-qual     → skip qualitative PDF plots

Outputs:
--------
- q1_agreement_heatmap.pdf → heatmap of method agreement
- qual_case_*.pdf          → 4 qualitative case plots (unless --skip-qual used)

How to run on a server (with nohup):
------------------------------------
# Activate your Python environment first, then run:

nohup python -u -m cli.plot_agreement_and_qual \
  --env-glob "results/envs/E_*.npz" \
  --outdir "runs/$(date +%Y%m%d_%H%M%S)/figs" \
  --planner a_star --connectivity 8 \
  --k 5 --max-envs 0 \
  --lime-samples 500 --lime-flip 0.30 \
  --shap-perm 100 \
  --workers 8 \
  --progress-every 5 \
  --heartbeat-secs 60 \
  > run.log 2>&1 &

# Monitor progress live:
tail -f run.log

# If you want to stop the job:
kill <PID>

Where <PID> can be retrieved with: ps -ef | grep plot_agreement_and_qual

Example quick run (for testing):
--------------------------------
python -m cli.plot_agreement_and_qual \
  --env-glob "results/envs/E_*.npz" \
  --outdir "runs/test/figs" \
  --planner a_star --connectivity 8 \
  --k 3 --max-envs 20 --skip-qual --workers 4

This will process only 20 environments, skip qualitative plots,
and use 4 workers for speed.
"""

from __future__ import annotations
import argparse
import glob
import os
import re
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

# --- ensure headless plotting on servers
import matplotlib
matplotlib.use("Agg")
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

METHODS = ["shap", "lime", "geodesic", "rand"]
M_INDEX = {m: i for i, m in enumerate(METHODS)}


def _load_env(npz_path: str):
    """Reconstruct a minimal GridEnvironment-like object from snapshot."""
    Z = np.load(npz_path, allow_pickle=True)

    class _Env:  # lightweight
        pass
    env = _Env()

    # Grid
    G = Z["grid"]
    env.grid = np.asarray(G, dtype=bool)

    # Object labels: keep as the original 2D int array (IDs: 0=free, 1..K=obstacles)
    if "obj_map" in Z:
        env.obj_map = np.asarray(Z["obj_map"])
    else:
        # fallback: derive labels from grid (each obstacle cell as its own ID)
        env.obj_map = np.zeros_like(env.grid, dtype=int)
        env.obj_map[env.grid] = np.arange(1, env.grid.sum() + 1)

    env.start = tuple(np.asarray(Z["start"]).tolist())
    env.goal  = tuple(np.asarray(Z["goal"]).tolist())
    env.H, env.W = env.grid.shape

    # obstacles list: 1..K (IDs present in obj_map)
    ids = np.unique(env.obj_map)
    env.obstacles = [i for i in ids.tolist() if i > 0]
    return env


def _parse_env_key_from_path(path: str) -> str:
    b = os.path.basename(path)
    m = re.match(r"E_(\d+)x(\d+)_d([0-9.]+)_([0-9]+)\.npz", b)
    if not m:
        return b
    H, W, d, eid = m.groups()
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
    ax.imshow(~env.grid, cmap="gray", interpolation="none")  # free=white, obstacles=black
    sr, sc = env.start; gr, gc = env.goal
    ax.scatter([sc], [sr], marker="o", s=60, edgecolors="k", facecolors="none", linewidths=2, label="Start")
    ax.scatter([gc], [gr], marker="*", s=80, edgecolors="k", facecolors="yellow", linewidths=1.5, label="Goal")
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


# ---------- Parallel worker ----------
def _init_planner(planner_name: str, connectivity: int):
    # re-create planner in each worker to avoid cross-process state issues
    return PLANNERS[planner_name](connectivity=connectivity)


def _process_one(args_tuple):
    """
    Worker: returns (ok_processed, considered_failure, M_add, C_add, elapsed_sec, env_key)
    where:
      M_add: float32 [len(METHODS), len(METHODS)]
      C_add: int32   [len(METHODS), len(METHODS)]
    """
    (npz_path, planner_name, connectivity, k, lime_ns, lime_fp, shap_perm, seed) = args_tuple
    t0 = time.time()
    try:
        env = _load_env(npz_path)
        planner = _init_planner(planner_name, connectivity)
        plan_ok = planner.plan(env.grid, env.start, env.goal)
        plan_ok = bool(plan_ok["success"] if isinstance(plan_ok, dict) else plan_ok)
        if plan_ok:
            # not a failure -> skip
            return (True, False, np.zeros((len(METHODS), len(METHODS)), dtype=np.float32),
                    np.zeros((len(METHODS), len(METHODS)), dtype=np.int32),
                    time.time() - t0, _parse_env_key_from_path(npz_path))

        sets = {}
        for m in METHODS:
            try:
                s = _ranking_topk(env, planner, m, k, lime_ns, lime_fp, shap_perm, seed+1)
                sets[m] = set(s)
            except Exception:
                # leave it out silently; pair contributions will be zero
                pass

        M_add = np.zeros((len(METHODS), len(METHODS)), dtype=np.float32)
        C_add = np.zeros((len(METHODS), len(METHODS)), dtype=np.int32)
        for a in METHODS:
            if a not in sets: 
                continue
            for b in METHODS:
                if a == b or b not in sets:
                    continue
                ia, ib = M_INDEX[a], M_INDEX[b]
                jac = 0.0
                # safe jaccard
                A, B = sets[a], sets[b]
                if A or B:
                    inter = len(A & B)
                    union = len(A | B)
                    jac = (inter / union) if union > 0 else 0.0
                M_add[ia, ib] += jac
                C_add[ia, ib] += 1

        return (True, True, M_add, C_add, time.time() - t0, _parse_env_key_from_path(npz_path))
    except Exception:
        return (False, False, np.zeros((len(METHODS), len(METHODS)), dtype=np.float32),
                np.zeros((len(METHODS), len(METHODS)), dtype=np.int32),
                time.time() - t0, _parse_env_key_from_path(npz_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-glob", type=str, default="results/envs/E_*.npz")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--planner", type=str, default="a_star", choices=list(PLANNERS.keys()))
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-envs", type=int, default=200, help="Cap total env files to scan")
    ap.add_argument("--max-failures", type=int, default=0, help="If >0, stop after this many failure envs considered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lime-samples", type=int, default=500)
    ap.add_argument("--lime-flip", type=float, default=0.30)
    ap.add_argument("--shap-perm", type=int, default=100)
    ap.add_argument("--qual-env", type=str, default="", help='Optional explicit key "HxW:density:env_id"')
    ap.add_argument("--skip-qual", action="store_true", help="Skip the 4 qualitative PDFs")
    # server-friendly knobs
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--progress-every", type=int, default=5, help="Print progress every N processed env files")
    ap.add_argument("--heartbeat-secs", type=int, default=60, help="Also print heartbeat every S seconds")
    args = ap.parse_args()

    np.random.seed(args.seed)
    _ensure_dir(args.outdir)

    # planner name used inside workers
    planner_name = args.planner

    # gather env paths
    paths = sorted(glob.glob(args.env_glob))
    if not paths:
        raise SystemExit(f"No env files for pattern: {args.env_glob}")
    if args.max_envs > 0:
        paths = paths[: args.max_envs]

    print(f"[setup] Found {len(paths)} env file(s). workers={args.workers}, planner={planner_name}, k={args.k}", flush=True)

    # -------- Parallel agreement computation -------- #
    from multiprocessing import Pool

    total = len(paths)
    processed = 0
    considered_fail = 0
    success_files = 0
    start = time.time()
    last_heartbeat = start
    sum_M = np.zeros((len(METHODS), len(METHODS)), dtype=np.float64)
    sum_C = np.zeros((len(METHODS), len(METHODS)), dtype=np.int64)

    def _progress_print(force=False):
        nonlocal last_heartbeat
        now = time.time()
        if (processed % max(1, args.progress_every) == 0) or force or (now - last_heartbeat >= args.heartbeat_secs):
            elapsed = now - start
            per = processed / total if total > 0 else 0.0
            eta = (elapsed / max(1, processed)) * (total - processed) if processed > 0 else 0.0
            print(f"[progress] {processed}/{total} files | failures considered={considered_fail} | "
                  f"ok_files={success_files} | elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                  flush=True)
            last_heartbeat = now

    # Build argument tuples once
    worker_args = [
        (p, planner_name, args.connectivity, args.k,
         args.lime_samples, args.lime_flip, args.shap_perm, args.seed)
        for p in paths
    ]

    # Use imap_unordered for streaming results
    with Pool(processes=args.workers, maxtasksperchild=64) as pool:
        for ok_processed, considered, M_add, C_add, sec, key in pool.imap_unordered(_process_one, worker_args, chunksize=1):
            processed += 1
            if ok_processed:
                success_files += 1
            if considered:
                considered_fail += 1
                sum_M += M_add
                sum_C += C_add

            if args.max_failures > 0 and considered_fail >= args.max_failures:
                print(f"[early-stop] Reached --max-failures={args.max_failures} after processing {processed} files.", flush=True)
                break

            if (processed % args.progress_every) == 0:
                _progress_print()

            # heartbeat based on time
            if (time.time() - last_heartbeat) >= args.heartbeat_secs:
                _progress_print(force=True)

    # Final progress line
    _progress_print(force=True)

    # average where count > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        M_avg = np.where(sum_C > 0, sum_M / np.maximum(1, sum_C), 0.0)

    # pandas for plotting (keeps same axis labels)
    M_df = pd.DataFrame(M_avg, index=METHODS, columns=METHODS)

    # plot heatmap
    fig = plt.figure(figsize=(5.6, 4.8))
    im = plt.imshow(M_df.values, origin="upper", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Mean Jaccard (top-k)")
    plt.xticks(np.arange(len(METHODS)), [m.upper() for m in METHODS], rotation=0)
    plt.yticks(np.arange(len(METHODS)), [m.upper() for m in METHODS])
    # annotate
    for i, a in enumerate(METHODS):
        for j, b in enumerate(METHODS):
            if i == j: 
                continue
            v = M_df.values[i, j]
            plt.text(j, i, f"{100*v:.0f}%", ha="center", va="center", fontsize=9)
    plt.title(f"Q1 Agreement Heatmap (planner={planner_name}, k={args.k})")
    plt.tight_layout()
    out_heat = os.path.join(args.outdir, "q1_agreement_heatmap.pdf")
    plt.savefig(out_heat, bbox_inches="tight"); plt.close()
    print(f"[q1_agreement] Wrote {out_heat}", flush=True)

    # --------- Qualitative case study (optional) --------- #
    if not args.skip_qual:
        # Pick env
        qual_path: Optional[str] = None
        if args.qual_env:
            for p in sorted(glob.glob(args.env_glob)):
                if _parse_env_key_from_path(p) == args.qual_env:
                    qual_path = p; break
            if qual_path is None:
                raise SystemExit(f"Could not find --qual-env '{args.qual_env}' in {args.env_glob}")
        else:
            qual_path = sorted(glob.glob(args.env_glob))[0]
        env = _load_env(qual_path)
        planner = _init_planner(planner_name, args.connectivity)

        # Base original
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _draw_grid(env, ax, title="Original (failure case)")
        out = os.path.join(args.outdir, "qual_case_original.pdf")
        plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
        print(f"[qual] Wrote {out}", flush=True)

        # SHAP
        shap = ShapExplainer(permutations=args.shap_perm, random_state=args.seed+11)
        shap_r = shap.explain(env, planner)["ranking"]
        shap_top = topk_set(shap_r, args.k)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _draw_grid(env, ax, title=f"SHAP (top-{args.k})", highlight=set(shap_top))
        out = os.path.join(args.outdir, "qual_case_shap.pdf")
        plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
        print(f"[qual] Wrote {out}", flush=True)

        # LIME
        lime = LimeExplainer(num_samples=args.lime_samples, flip_prob=args.lime_flip, random_state=args.seed+13)
        lime_r = lime.explain(env, planner)["ranking"]
        lime_top = topk_set(lime_r, args.k)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _draw_grid(env, ax, title=f"LIME (top-{args.k})", highlight=set(lime_top))
        out = os.path.join(args.outdir, "qual_case_lime.pdf")
        plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
        print(f"[qual] Wrote {out}", flush=True)

        # COSE (guided by SHAP)
        cose = COSEExplainer(guide="shap")
        cose_out = cose.explain(env, planner, guide_ranking=shap_r)
        cose_set = set(cose_out.get("cose_set", []))
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _draw_grid(env, ax, title="COSE (counterfactual set)", highlight=cose_set)
        out = os.path.join(args.outdir, "qual_case_cose.pdf")
        plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()
        print(f"[qual] Wrote {out}", flush=True)
    else:
        print("[qual] Skipped qualitative plots (--skip-qual).", flush=True)

    total_elapsed = time.time() - start
    print(f"[done] processed_files={processed} | failures_considered={considered_fail} | "
          f"outdir={args.outdir} | total_elapsed={total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
