# explanations/eval/__init__.py
# -*- coding: utf-8 -*-
"""
Evaluation utilities: metrics (always), and optional ILP / transfer / robustness.
This avoids import-time hard failures when optional deps aren't installed.
"""

from __future__ import annotations

# --- Always available: metrics ---
from .metrics import (
    success_at_k,
    auc_success_at_k,
    topk_set,
    jaccard,
    kendall_tau,
    evaluate_ranking_success_curve,
    sum_calls,
    sum_time_sec,
)

# --- Optional: ILP oracle (PuLP) ---
try:
    from .ilp_minimality import exact_minimal_set  # noqa: F401
except Exception:
    def exact_minimal_set(*args, **kwargs):
        raise RuntimeError(
            "exact_minimal_set is unavailable (PuLP not installed or solver missing). "
            "Install with `pip install pulp` (and optionally CBC)."
        )

# --- Optional: transfer (pure-Python, but guard to avoid import loops) ---
try:
    from .transfer import cross_planner_success_at_k, cross_planner_overlap  # noqa: F401
except Exception:
    def cross_planner_success_at_k(*args, **kwargs):
        raise RuntimeError("cross_planner_success_at_k unavailable (transfer module failed to import).")
    def cross_planner_overlap(*args, **kwargs):
        raise RuntimeError("cross_planner_overlap unavailable (transfer module failed to import).")

# --- Optional: robustness (SciPy) ---
try:
    from .robustness import (  # noqa: F401
        robustness_suite,
        perturb_jitter,
        perturb_dilate,
        perturb_erode,
        perturb_distractors,
    )
except Exception:
    def robustness_suite(*args, **kwargs):
        raise RuntimeError("robustness_suite unavailable (SciPy not installed). Install with `pip install scipy`.")
    def perturb_jitter(*args, **kwargs): return robustness_suite(*args, **kwargs)
    def perturb_dilate(*args, **kwargs): return robustness_suite(*args, **kwargs)
    def perturb_erode(*args, **kwargs): return robustness_suite(*args, **kwargs)
    def perturb_distractors(*args, **kwargs): return robustness_suite(*args, **kwargs)

__all__ = [
    # metrics (always)
    "success_at_k", "auc_success_at_k", "topk_set", "jaccard", "kendall_tau",
    "evaluate_ranking_success_curve", "sum_calls", "sum_time_sec",
    # ilp (optional)
    "exact_minimal_set",
    # transfer (optional)
    "cross_planner_success_at_k", "cross_planner_overlap",
    # robustness (optional)
    "robustness_suite", "perturb_jitter", "perturb_dilate", "perturb_erode", "perturb_distractors",
]
