#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test package for the environment_centered_explanations project.

This file ensures the 'tests' directory is recognized as a package
so relative imports work (e.g., from envs import generator).
"""

import os
import sys

# Automatically add the project root to sys.path when running tests directly
# (so you can run pytest from anywhere, e.g., `pytest tests/`)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Optional: silence common numpy warnings during tests (optional)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
