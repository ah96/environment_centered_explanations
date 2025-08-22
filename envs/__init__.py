# -*- coding: utf-8 -*-
"""
Environment generation and affordances.
Exposes:
- GridEnvironment (dataclass from generator.py)
- generate_environment(...)
- rebuild_objects_from_grid(...)  (if defined in generator.py)
- move/remove affordances (from affordances.py)
"""

from __future__ import annotations

# Prefer explicit imports; if any symbol is missing in your local generator/affordances,
# simply remove it from __all__ below.
from .generator import GridEnvironment, generate_environment
try:
    from .generator import rebuild_objects_from_grid  # optional utility
except Exception:  # noqa: BLE001
    rebuild_objects_from_grid = None  # type: ignore

# Common affordances (may not be used directly by CLIs but handy for users)
# envs/__init__.py  (replace the affordances import block)
try:
    from .affordances import move_obstacle, remove_obstacles as remove_obstacle
except Exception:  # noqa: BLE001
    move_obstacle = None  # type: ignore
    remove_obstacle = None  # type: ignore

__all__ = [
    "GridEnvironment",
    "generate_environment",
    "rebuild_objects_from_grid",
    "move_obstacle",
    "remove_obstacle",  # alias to remove_obstacles
]

