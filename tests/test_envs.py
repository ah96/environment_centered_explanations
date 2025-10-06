#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.generator import generate_environment, _free_bfs_has_path, active_obstacles, _dilate_bool
from envs.affordances import remove_obstacles, move_obstacle

def test_generate_failure_has_no_path():
    rng = np.random.default_rng(123)
    env = generate_environment(H=40, W=40, density=0.2, ensure_status="failure", rng=rng)
    assert not _free_bfs_has_path(env.grid, env.start, env.goal)

def test_remove_marks_empties_and_preserves_start_goal():
    rng = np.random.default_rng(0)
    env = generate_environment(H=40, W=40, density=0.2, ensure_status="failure", rng=rng)
    if env.obstacles:
        smallest = int(np.argmin([ob.area for ob in env.obstacles])) + 1
        env2 = remove_obstacles(env, [smallest], inplace=False, relabel_after=False)
        # obstacle becomes empty, but id remains
        ob = env2.obstacles[smallest-1]
        assert ob.coords.size == 0
        # start/goal remain free
        assert env2.grid[env2.start] == False
        assert env2.grid[env2.goal] == False

def test_move_respects_moat_and_id_stability():
    rng = np.random.default_rng(7)
    env = generate_environment(H=50, W=50, density=0.18, ensure_status="any", rng=rng)
    oid = int(rng.integers(1, len(env.obstacles)+1))
    env2, moved = move_obstacle(env, oid, max_translation=6, n_candidates=120, moat=1)
    assert moved
    # moat check: dilate moved object and ensure ring doesn't touch other obstacles
    ob = env2.obstacles[oid-1]
    obj_mask = np.zeros(env2.shape, dtype=bool)
    obj_mask[ob.coords[:,0], ob.coords[:,1]] = True
    ring = _dilate_bool(obj_mask, 1)
    grid_wo_obj = env2.grid.copy()
    grid_wo_obj[ob.coords[:,0], ob.coords[:,1]] = False
    assert not (ring & grid_wo_obj).any()
    # id preserved in obj_map
    assert (env2.obj_map[ob.coords[:,0], ob.coords[:,1]] == oid).all()

def test_relabeling_compacts_ids_when_requested():
    rng = np.random.default_rng(1)
    env = generate_environment(H=40, W=40, density=0.2, ensure_status="any", rng=rng)
    # remove a few, then relabel
    ids = [1, 3, 5][:min(3, len(env.obstacles))]
    env2 = remove_obstacles(env, ids, inplace=False, relabel_after=True)
    # After relabel, obj_map labels are 1..K with no holes
    labels = np.unique(env2.obj_map)
    labels = labels[labels > 0]
    if labels.size:
        assert labels.min() == 1
        assert labels.max() == len(env2.obstacles)  # cc_label guarantees compactness here
