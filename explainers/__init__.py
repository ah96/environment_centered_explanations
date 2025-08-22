# -*- coding: utf-8 -*-
"""
Explainers for environment-centered attribution/counterfactuals.

Conventions
-----------
- Ranking explainers return {'ranking': [(obs_id, score), ...], 'calls', 'time_sec', ...}
  (LIME/SHAP/baselines); higher score means *more harmful* (more responsible for failure).
- COSE returns {'cose_set': set[int], 'calls', 'time_sec', ...}
"""

from __future__ import annotations
from typing import Dict, Any, Callable

from .lime_explainer import LimeExplainer
from .shap_explainer import ShapExplainer
from .cose import COSEExplainer
from .baselines import random_ranking, geodesic_line_ranking

# Factory mapping for top-level convenience (explanations.get_explainer)
# Note: baselines are callables, not classes; they are included directly.
EXPLAINERS: Dict[str, Any] = {
    "lime": LimeExplainer,
    "shap": ShapExplainer,
    "cose": COSEExplainer,
    "rand": random_ranking,
    "geodesic": geodesic_line_ranking,
}

__all__ = [
    "LimeExplainer",
    "ShapExplainer",
    "COSEExplainer",
    "random_ranking",
    "geodesic_line_ranking",
    "EXPLAINERS",
]
