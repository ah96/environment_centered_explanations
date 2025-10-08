#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIME-style local surrogate explainer for obstacle-level attribution.

This is a lightweight, dependency-free adaptation of Kernel LIME for binary
features (obstacle presence). It fits a *weighted ridge regression* on
synthetic perturbations around the current environment and interprets the
coefficients as local importance scores.

Key choices here are tailored for grid navigation:
- Features z \in {0,1}^m (1 = obstacle present).
- Proximity kernel = exp(-(d_H(z, z0)/m)^2 / sigma^2), where z0 = all-ones.
- Labels are binary {0,1}: planner success (1) or failure (0).
- We report "harmfulness" = -coef, so higher score => obstacle more *responsible for failure*.

Returns:
    {
      'ranking': [(obs_id, score), ...]  # descending by score (harmfulness)
      'weights': np.ndarray[m],          # raw coefficients (positive helps success)
      'calls': int,
      'time_sec': float
    }
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import time
from explainers.baselines import geodesic_line_ranking


class LimeExplainer:
    """
    LIME for binary obstacle presence with Hamming RBF kernel + ridge.
    Optional focus_top_m: restrict perturbations to top-M obstacles from a cheap heuristic.
    """
    def __init__(self, num_samples: int = 500, flip_prob: float = 0.3,
                 random_state: int = None, alpha: float = 1e-2,
                 focus_top_m: int = None):
        self.num_samples = int(num_samples)
        self.flip_prob = float(flip_prob)
        self.rng = np.random.default_rng(random_state)
        self.alpha = float(alpha)
        self.focus_top_m = int(focus_top_m) if focus_top_m else None

    def explain(self, env, planner):
        t0 = time.perf_counter()
        n = len(env.obstacles)
        all_ids = np.array([i for i,o in enumerate(env.obstacles, start=1) if getattr(o, "coords", None) is not None and o.coords.size > 0], dtype=int)

        # Handle edge case: no valid obstacles
        if len(all_ids) == 0:
            return {
                "ranking": [],
                "calls": 0,
                "time_sec": time.perf_counter() - t0,
                "considered": 0,
                "n_total": n,
                "focus_top_m": self.focus_top_m or 0,
            }

        # ---- Focus subset (optional)
        if self.focus_top_m and self.focus_top_m < len(all_ids):
            geo = geodesic_line_ranking(env)["ranking"]
            subset_ids = np.array([oid for oid, _ in geo[: self.focus_top_m]], dtype=int)
        else:
            subset_ids = all_ids

        m = len(subset_ids)
        if m == 0:
            return {
                "ranking": [],
                "calls": 0,
                "time_sec": time.perf_counter() - t0,
                "considered": 0,
                "n_total": n,
                "focus_top_m": self.focus_top_m or 0,
            }

        id_from_local = subset_ids
        obj_map = env.obj_map
        base_grid = env.grid.copy()

        # Sample Z in {0,1}^m, where 1 = obstacle present (relative to original)
        Z = self.rng.binomial(1, 1.0 - self.flip_prob, size=(self.num_samples, m)).astype(bool)
        # ensure the original point appears at least once (all 1s)
        if self.num_samples > 0:
            Z[0, :] = True

        # Evaluate planner on each perturbation
        Y = np.empty(self.num_samples, dtype=float)
        calls = 0
        for i in range(self.num_samples):
            grid = base_grid.copy()
            # turn OFF any subset obstacle with Z[i,j] == 0
            for j in range(m):
                if not Z[i, j]:
                    oid = int(id_from_local[j])
                    grid[obj_map == oid] = False
            res = planner.plan(grid, env.start, env.goal)
            Y[i] = 1.0 if (res.get("success", False) if isinstance(res, dict) else bool(res)) else 0.0
            calls += 1

        # Check if all outcomes are the same (no variance to explain)
        if np.all(Y == Y[0]):
            # Assign uniform scores
            harm_local = np.zeros(m)
            pairs = [(int(id_from_local[j]), 0.0) for j in range(m)]
            if m < len(all_ids):
                for oid in all_ids:
                    if oid not in id_from_local:
                        pairs.append((int(oid), 0.0))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return {
                "ranking": pairs,
                "calls": calls,
                "time_sec": time.perf_counter() - t0,
                "considered": int(m),
                "n_total": int(n),
                "focus_top_m": self.focus_top_m or 0,
            }

        # Hamming distance kernel around the origin (all ones)
        # distance = number of flips from the original
        d = (1.0 - Z).sum(axis=1).astype(float)
        # RBF on Hamming with bandwidth = max(1, m/4) (can be tuned)
        sigma = max(1.0, m / 4.0)
        W = np.exp(-(d ** 2) / (2.0 * sigma * sigma))

        # Weighted ridge solve for linear model: Y ≈ w0 + sum_j w_j Z_j
        # augment Z with bias column
        X = np.hstack([np.ones((self.num_samples, 1)), Z.astype(float)])
        # Closed-form (X^T W X + αI)^{-1} X^T W Y
        Wdiag = np.diag(W)
        XtW = X.T @ Wdiag
        A = XtW @ X
        # ridge on all coefficients (including bias) for stability
        A += self.alpha * np.eye(A.shape[0])
        b = XtW @ Y
        coef = np.linalg.solve(A, b)  # [w0, w1..wm]
        w = coef[1:]  # per-feature weights

        # Harmfulness: negative weight (obstacle presence decreasing success ⇒ harmful)
        harm_local = -w

        pairs = [(int(id_from_local[j]), float(harm_local[j])) for j in range(m)]
        if m < len(all_ids):
            floor = (min(harm_local) if m > 0 else 0.0) - 1e6
            for oid in all_ids:
                if oid not in id_from_local:
                    pairs.append((int(oid), float(floor) - 1e-3 * int(oid)))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return {
            "ranking": pairs,
            "calls": calls,
            "time_sec": time.perf_counter() - t0,
            "considered": int(m),
            "n_total": int(n),
            "focus_top_m": self.focus_top_m or 0,
        }