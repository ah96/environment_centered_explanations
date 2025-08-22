#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figs.py
------------
Reads CSV outputs from the CLI runs and generates publication-ready plots
(using matplotlib only) with filenames matching your LaTeX.

Generates (if corresponding CSVs are found):
  - figs/q1_success_at_k.pdf
  - figs/q2_min_set_size_violin.pdf   (boxplot approximation)
  - figs/q2_optimality_gap_bar.pdf
  - figs/q3_robustness_kendall_bar.pdf
  - figs/q3_cross_planner_heatmap.pdf

Notes
-----
- No seaborn; pure matplotlib.
- Lines/bars use default colors; single plot per figure.
- If a required CSV is missing for a figure, that figure is skipped gracefully.
"""

from __future__ import annotations
import argparse
import ast
import glob
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- I/O helpers ------------------------- #

def _load_many(glob_pat: str) -> Optional[pd.DataFrame]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        print(f"[make_figs] No files for pattern: {glob_pat}")
        return None
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src__"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[make_figs] Skipping {p}: {e}")
    if not dfs:
        return None
    out = pd.concat(dfs, ignore_index=True)
    print(f"[make_figs] Loaded {len(paths)} file(s) → {len(out)} rows from {glob_pat}")
    return out


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------- parsing helpers ------------------------- #

def _parse_success_at_k_dict(x) -> Dict[int, int]:
    """
    success_at_k is stored as a string repr of a dict, e.g., "{1: 0, 2: 1, 3: 1}".
    """
    if isinstance(x, dict):
        return {int(k): int(v) for k, v in x.items()}
    try:
        d = ast.literal_eval(str(x))
        if isinstance(d, dict):
            return {int(k): int(v) for k, v in d.items()}
    except Exception:
        pass
    return {}


def _min_k_success(succ_dict: Dict[int, int], kmax: int) -> Optional[int]:
    """
    Return smallest k in [1..kmax] where success@k==1, else None.
    """
    for k in range(1, int(kmax) + 1):
        if int(succ_dict.get(k, 0)) == 1:
            return k
    return None


# ----------------------------- Figure 1: Q1 Success@k ----------------------------- #

def plot_q1_success_at_k(eval_df: pd.DataFrame, outpath: str):
    """
    Aggregate mean Success@k across environments/planners for ranking methods only.
    Methods considered: shap, lime, rand, geodesic (skip cose; it's set-based).
    """
    df = eval_df.copy()
    df = df[df["method"].isin(["shap", "lime", "rand", "geodesic"])]
    if df.empty:
        print("[q1_success_at_k] No ranking methods in eval CSV. Skipping.")
        return

    # Expand success_at_k dicts into rows (method, k, success)
    rows = []
    for _, r in df.iterrows():
        succ = _parse_success_at_k_dict(r.get("success_at_k", "{}"))
        for k, v in succ.items():
            rows.append({
                "method": r["method"],
                "planner": r["planner"],
                "k": int(k),
                "success": int(v)
            })
    if not rows:
        print("[q1_success_at_k] Could not parse success_at_k. Skipping.")
        return
    E = pd.DataFrame(rows)

    # Mean Success@k over planners and envs
    G = E.groupby(["method", "k"])["success"].mean().reset_index()

    # Plot
    plt.figure(figsize=(6.0, 4.0))
    methods = ["shap", "lime", "geodesic", "rand"]
    for m in methods:
        g = G[G["method"] == m]
        if g.empty:
            continue
        ks = g["k"].values
        ys = g["success"].values
        plt.plot(ks, ys, marker="o", label=m.upper())

    plt.xlabel("k (top-k obstacles removed)")
    plt.ylabel("Success@k (mean)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[q1_success_at_k] Wrote {outpath}")


# ----------------------------- Figure 3: Q2 set-size distribution ----------------------------- #

def plot_q2_min_set_size(eval_df: pd.DataFrame, outpath: str):
    """
    Boxplot of explanation set sizes:
      - For ranking methods, derive minimal k where success@k==1.
      - For COSE, use expl_set_size column directly.
    """
    df = eval_df.copy()
    if df.empty:
        print("[q2_min_set_size] Empty eval CSV. Skipping.")
        return

    K_by_row = {}
    if "kmax" in df.columns:
        K_by_row = df["kmax"].to_dict()

    sizes = {}
    # Ranking methods → minimal k
    for m in ["shap", "lime", "geodesic", "rand"]:
        d = df[df["method"] == m]
        vals = []
        for idx, r in d.iterrows():
            succ = _parse_success_at_k_dict(r.get("success_at_k", "{}"))
            kmax = int(r.get("kmax", 0) or 0)
            if kmax <= 0:
                continue
            mk = _min_k_success(succ, kmax)
            if mk is not None:
                vals.append(int(mk))
        if vals:
            sizes[m.upper() + "-TOP-K"] = vals

    # COSE → expl_set_size
    d = df[df["method"] == "cose"]
    vals = []
    for _, r in d.iterrows():
        v = r.get("expl_set_size", None)
        if pd.notna(v):
            try:
                vals.append(int(v))
            except Exception:
                pass
    if vals:
        sizes["COSE"] = vals

    if not sizes:
        print("[q2_min_set_size] No sizes to plot. Skipping.")
        return

    # Boxplot (violin-like summary)
    labels = list(sizes.keys())
    data = [sizes[k] for k in labels]

    plt.figure(figsize=(6.0, 4.0))
    bp = plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Explanation set size (lower is better)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[q2_min_set_size] Wrote {outpath}")


# ----------------------------- Figure 4: Q2 optimality gap (ILP) ----------------------------- #

def plot_q2_optimality_gap(exact_df: pd.DataFrame, outpath: str):
    """
    Bar chart of average optimality gap (COSE size - ILP size) per (H×W, density).
    """
    df = exact_df.copy()
    if df is None or df.empty:
        print("[q2_opt_gap] No exact_small CSV rows. Skipping.")
        return
    if "gap" not in df.columns:
        print("[q2_opt_gap] 'gap' column not found. Skipping.")
        return

    # Group by setting
    df["setting"] = df["H"].astype(str) + "x" + df["W"].astype(str) + " @ d=" + df["density"].astype(str)
    G = df.groupby("setting")["gap"].agg(["mean", "count", "std"]).reset_index()
    G["sem"] = G["std"] / np.sqrt(G["count"].clip(lower=1))
    x = np.arange(len(G))
    means = G["mean"].values
    sems = G["sem"].values
    labels = G["setting"].tolist()

    plt.figure(figsize=(7.0, 4.0))
    plt.bar(x, means, yerr=sems, capsize=4)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Optimality gap (COSE − ILP)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[q2_opt_gap] Wrote {outpath}")


# ----------------------------- Figure 5: Q3 robustness Kendall ----------------------------- #

def plot_q3_robustness_kendall(rob_df: pd.DataFrame, outpath: str):
    """
    Bar chart of average Kendall's tau across perturbations and planners for ranking methods.
    """
    df = rob_df.copy()
    if df is None or df.empty:
        print("[q3_robust_kendall] No robustness CSV rows. Skipping.")
        return

    d = df[df["method"].isin(["shap", "lime", "geodesic", "rand"])]
    if d.empty or "kendall_tau" not in d.columns:
        print("[q3_robust_kendall] No ranking methods or 'kendall_tau' missing. Skipping.")
        return

    G = d.groupby("method")["kendall_tau"].mean().reindex(["shap", "lime", "geodesic", "rand"])
    G = G.dropna()
    x = np.arange(len(G))
    plt.figure(figsize=(6.0, 4.0))
    plt.bar(x, G.values)
    plt.xticks(x, [m.upper() for m in G.index])
    plt.ylabel("Kendall's $\\tau$ (higher = more stable)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[q3_robust_kendall] Wrote {outpath}")


# ----------------------------- Figure 6: Q3 cross-planner heatmap ----------------------------- #

def plot_q3_cross_planner_heatmap(transfer_df: pd.DataFrame, outpath: str, method: str = "shap"):
    """
    Heatmap of mean overlap_topk (%) across planner pairs for a chosen method (default SHAP).
    """
    df = transfer_df.copy()
    if df is None or df.empty:
        print("[q3_cross_heatmap] No transfer CSV rows. Skipping.")
        return
    df = df[df["method"] == method]
    if df.empty or "overlap_topk" not in df.columns:
        print(f"[q3_cross_heatmap] No rows for method '{method}' or 'overlap_topk' missing. Skipping.")
        return

    # Build planner list and matrix
    planners = sorted(set(df["planner_A"]) | set(df["planner_B"]))
    idx = {p: i for i, p in enumerate(planners)}
    M = np.zeros((len(planners), len(planners)), dtype=float)
    C = np.zeros_like(M)

    for _, r in df.iterrows():
        a, b = r["planner_A"], r["planner_B"]
        if a == b:
            continue
        try:
            v = float(r["overlap_topk"])
        except Exception:
            continue
        ia, ib = idx[a], idx[b]
        M[ia, ib] += v
        C[ia, ib] += 1

    C[C == 0] = 1
    M = M / C

    plt.figure(figsize=(6.5, 5.2))
    im = plt.imshow(M, origin="upper", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Top-k overlap (Jaccard)")
    plt.xticks(np.arange(len(planners)), [p.upper() for p in planners], rotation=30, ha="right")
    plt.yticks(np.arange(len(planners)), [p.upper() for p in planners])
    # Annotate
    for i in range(len(planners)):
        for j in range(len(planners)):
            if i == j:
                continue
            plt.text(j, i, f"{M[i,j]*100:.0f}%", ha="center", va="center", fontsize=8)
    plt.title(f"Cross-planner stability ({method.upper()})")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[q3_cross_heatmap] Wrote {outpath}")


# ----------------------------- (Optional) Q1 agreement heatmap ----------------------------- #

def plot_q1_agreement_heatmap_placeholder(outpath: str):
    """
    Placeholder if you later export per-environment *method-to-method* overlaps.
    For now we skip because eval CSVs don't contain the actual top-k sets/ids.
    """
    print("[q1_agreement_heatmap] Skipped (requires per-method top-k ID sets).")


# ----------------------------- main ----------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Generate paper figures from CSV outputs.")
    ap.add_argument("--eval-glob", type=str, default="results/csv/eval_*.csv")
    ap.add_argument("--exact-glob", type=str, default="results/csv/exact_small_*.csv")
    ap.add_argument("--transfer-glob", type=str, default="results/csv/transfer_*.csv")
    ap.add_argument("--robustness-glob", type=str, default="results/csv/robustness_*.csv")
    ap.add_argument("--outdir", type=str, default="results/figs")
    ap.add_argument("--transfer-method", type=str, default="shap", choices=["shap","lime","geodesic","rand"])
    args = ap.parse_args()


    _ensure_dir(args.outdir)

    df_eval = _load_many(args.eval_glob)
    df_exact = _load_many(args.exact_glob)
    df_trans = _load_many(args.transfer_glob)
    df_rob = _load_many(args.robustness_glob)

    # Q1 Success@k
    if df_eval is not None:
        plot_q1_success_at_k(df_eval, os.path.join(args.outdir, "q1_success_at_k.pdf"))
    else:
        print("[main] Skipping Q1 Success@k (no eval data).")

    # Q2 min set size
    if df_eval is not None:
        plot_q2_min_set_size(df_eval, os.path.join(args.outdir, "q2_min_set_size_violin.pdf"))
    else:
        print("[main] Skipping Q2 set size (no eval data).")

    # Q2 optimality gap
    if df_exact is not None:
        plot_q2_optimality_gap(df_exact, os.path.join(args.outdir, "q2_optimality_gap_bar.pdf"))
    else:
        print("[main] Skipping Q2 optimality gap (no exact_small data).")

    # Q3 robustness Kendall
    if df_rob is not None:
        plot_q3_robustness_kendall(df_rob, os.path.join(args.outdir, "q3_robustness_kendall_bar.pdf"))
    else:
        print("[main] Skipping Q3 robustness (no robustness data).")

    # Q3 cross-planner heatmap
    if df_trans is not None:
        plot_q3_cross_planner_heatmap(df_trans, os.path.join(args.outdir, "q3_cross_planner_heatmap.pdf"),
                                      method=args.transfer_method)
    else:
        print("[main] Skipping cross-planner heatmap (no transfer data).")

    # Optional Q1 agreement heatmap (requires extra data not in current CSVs)
    # plot_q1_agreement_heatmap_placeholder(os.path.join(args.outdir, "q1_agreement_heatmap.pdf"))

    print(f"[DONE] Figures in: {args.outdir}")


if __name__ == "__main__":
    main()
