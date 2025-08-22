# -*- coding: utf-8 -*-
"""
Planners on grid maps with a unified API:
planner.plan(grid: np.ndarray[bool], start: (r,c), goal: (r,c))
  -> {'success': bool, 'path': List[(r,c)] or None}
"""

from __future__ import annotations
from typing import Dict, Type

from .a_star import AStarPlanner
from .dijkstra import DijkstraPlanner
from .bfs import BFSPlanner
from .dfs import DFSPlanner
from .theta_star import ThetaStarPlanner

# Mapping used by factories/CLIs
PLANNERS: Dict[str, Type] = {
    "a_star": AStarPlanner,
    "dijkstra": DijkstraPlanner,
    "bfs": BFSPlanner,
    "dfs": DFSPlanner,
    "theta_star": ThetaStarPlanner,
}

__all__ = [
    "AStarPlanner",
    "DijkstraPlanner",
    "BFSPlanner",
    "DFSPlanner",
    "ThetaStarPlanner",
    "PLANNERS",
]
