# -*- coding: utf-8 -*-
"""
Command-line entry points (run with `python -m explanations.cli.<name>`):

- run_eval          : main quantitative evaluation (Success@k, AUC-S@K, COSE sizes)
- run_exact_small   : ILP oracle vs COSE minimality on small grids
- run_transfer      : cross-planner transfer (apply A→B, compare overlaps)
- run_robustness    : perturbation robustness (Kendall/Jaccard)
- make_figs         : generate paper figures from CSVs
"""
__all__ = [
    "run_eval",
    "run_exact_small",
    "run_transfer",
    "run_robustness",
    "make_figs",
]
