import os
import json
from grid_world_env import GridWorldEnv
from path_planning import AStarPlanner

def save_env(env, path_A, path_B, index, output_dir="contrastive_envs_structured"):
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "grid_size": env.grid_size,
        "start": env.agent_pos,
        "goal": env.goal_pos,
        "obstacles": env.obstacles,
        "obstacle_shapes": env.obstacle_shapes,
        "path_A": path_A,
        "path_B": path_B
    }
    with open(os.path.join(output_dir, f"env_{index:03d}.json"), "w") as f:
        json.dump(data, f)

def generate_structured_env(index):
    grid_size = 10
    env = GridWorldEnv(grid_size=grid_size, num_obstacles=0)
    env.agent_pos = [0, 0]
    env.goal_pos = [grid_size - 1, grid_size - 1]

    # Create a fork in the middle with two paths
    upper_wall = [(4, y) for y in range(1, 9)]
    lower_wall = [(5, y) for y in range(1, 9)]

    sid_upper = env.add_obstacle_shape(upper_wall)
    sid_lower = env.add_obstacle_shape(lower_wall)

    planner = AStarPlanner()
    planner.set_environment(env.agent_pos, env.goal_pos, env.grid_size, env.obstacles)
    path_A = planner.plan()

    if not path_A:
        print(f"[✗] Initial path_A failed for env {index:03d}")
        return False

    # Create env variant with upper wall removed
    env_variant = env.clone()
    env_variant.remove_obstacle_shape(sid_upper)

    planner = AStarPlanner()
    planner.set_environment(env_variant.agent_pos, env_variant.goal_pos, env_variant.grid_size, env_variant.obstacles)
    path_B = planner.plan()

    if not path_B or path_A == path_B:
        print(f"[✗] Failed to generate contrastive path_B for env {index:03d}")
        return False

    save_env(env, path_A, path_B, index)
    print(f"[✓] Saved structured env_{index:03d}.json")
    return True

if __name__ == "__main__":
    for i in range(10):
        generate_structured_env(i)
