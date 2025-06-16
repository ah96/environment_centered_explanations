import os
import json
import random
from grid_world_env import GridWorldEnv
from path_planning import AStarPlanner


def save_env(env, path_A, path_B, modification, index, output_dir="longer_path_envs"):
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "grid_size": env.grid_size,
        "start": env.agent_pos,
        "goal": env.goal_pos,
        "obstacles": list(env.obstacles),
        "obstacle_shapes": env.obstacle_shapes,
        "path_A": path_A,
        "path_B": path_B,
        "modification": modification
    }
    with open(os.path.join(output_dir, f"env_{index:03d}.json"), "w") as f:
        json.dump(data, f)


def shape_close_to_path(shape_cells, path, threshold=2):
    for (x1, y1) in shape_cells:
        for (x2, y2) in path:
            if abs(x1 - x2) + abs(y1 - y2) <= threshold:
                return True
    return False


def all_possible_perturbations(original_cells, grid_size):
    perturbations = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            new_cells = [(x + dx, y + dy) for (x, y) in original_cells]
            if all(0 <= x < grid_size and 0 <= y < grid_size for (x, y) in new_cells):
                perturbations.append((dx, dy, new_cells))
    return perturbations


def generate_env(index, max_attempts=200, grid_size=10):
    for attempt in range(max_attempts):
        env = GridWorldEnv(grid_size=grid_size, num_obstacles=15)
        env.agent_pos = [0, 0]
        env.goal_pos = [grid_size - 1, grid_size - 1]

        planner = AStarPlanner()
        planner.set_environment(env.agent_pos, env.goal_pos, env.grid_size, env.obstacles)
        path_A = planner.plan()
        if not path_A:
            continue

        shape_ids = [sid for sid, cells in env.obstacle_shapes.items()
                     if shape_close_to_path(cells, path_A)]
        random.shuffle(shape_ids)

        for sid in shape_ids:
            original_cells = env.obstacle_shapes[sid]

            perturbations = all_possible_perturbations(original_cells, grid_size)
            for dx, dy, new_cells in perturbations:
                for removal_first in [True, False]:
                    test_env = env.clone()

                    if removal_first:
                        test_env.remove_obstacle_shape(sid)
                        new_sid = max(test_env.obstacle_shapes.keys(), default=0) + 1
                        test_env.obstacle_shapes[new_sid] = new_cells
                        test_env.obstacles.update(new_cells)
                    else:
                        # Add new shape, then remove old one
                        new_sid = max(test_env.obstacle_shapes.keys(), default=0) + 1
                        test_env.obstacle_shapes[new_sid] = new_cells
                        test_env.obstacles.update(new_cells)
                        test_env.remove_obstacle_shape(sid)

                    planner = AStarPlanner()
                    planner.set_environment(test_env.agent_pos, test_env.goal_pos, test_env.grid_size, test_env.obstacles)
                    path_B = planner.plan()
                    if path_B and len(path_B) < len(path_A) - 1:
                        save_env(env, path_A, path_B,
                                 {"action": "moved+removed", "from": original_cells, "to": new_cells, "offset": [dx, dy]}, index)
                        return True

            # Try removal only
            test_env = env.clone()
            test_env.remove_obstacle_shape(sid)
            planner = AStarPlanner()
            planner.set_environment(test_env.agent_pos, test_env.goal_pos, test_env.grid_size, test_env.obstacles)
            path_B = planner.plan()
            if path_B and len(path_B) < len(path_A) - 1:
                save_env(env, path_A, path_B, {"action": "removed", "sid": sid}, index)
                return True

    print(f"[✗] Failed to create env {index:03d}")
    return False


if __name__ == "__main__":
    success = 0
    for i in range(100):
        if generate_env(i):
            print(f"[✓] Created env {i:03d}")
            success += 1
        else:
            print(f"[✗] Failed env {i:03d}")
    print(f"\nTotal successful environments: {success}/100")
