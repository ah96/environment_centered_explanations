# -*- coding: utf-8 -*-
"""
Top-level package for environment-centered planning explanations.
Provides convenience factories for planners and explainers.
"""

from __future__ import annotations
from typing import Any, Dict, Type, Optional

__all__ = [
    "__version__",
    "get_planner",
    "get_explainer",
]

__version__ = "0.1.0"


def get_planner(name: str, **kwargs) -> Any:
    """
    Factory: instantiate a planner by name.

    Parameters
    ----------
    name : str
        One of: 'a_star', 'dijkstra', 'bfs', 'dfs', 'theta_star'
    kwargs : dict
        Passed to the planner constructor (e.g., connectivity=8)

    Returns
    -------
    planner instance
    """
    name = name.strip().lower()
    from .planners import PLANNERS  # lazy import
    if name not in PLANNERS:
        raise ValueError(f"Unknown planner '{name}'. Available: {sorted(PLANNERS)}")
    return PLANNERS[name](**kwargs)


def get_explainer(name: str, **kwargs) -> Any:
    """
    Factory: instantiate an explainer by name.

    Parameters
    ----------
    name : str
        One of: 'lime', 'shap', 'cose', 'rand', 'geodesic'
    kwargs : dict
        Constructor args (if any). For baselines, kwargs are ignored.

    Returns
    -------
    explainer instance OR callable returning a dict for baselines.
    """
    name = name.strip().lower()
    from .explainers import EXPLAINERS  # lazy import
    if name not in EXPLAINERS:
        raise ValueError(f"Unknown explainer '{name}'. Available: {sorted(EXPLAINERS)}")
    factory = EXPLAINERS[name]
    return factory(**kwargs) if callable(factory) and getattr(factory, "__name__", "") not in ("random_ranking", "geodesic_line_ranking") else factory
