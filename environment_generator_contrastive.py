import os
import json
import random
from grid_world_env import GridWorldEnv
from path_planning.astar import AStarPlanner

class ContrastiveEnvironmentGenerator:
    def __init__(self, grid_size=10, num_obstacles=6, seed=42):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.rng = random.Random(seed)  # Use local RNG
        os.makedirs("contrastive_envs", exist_ok=True)

    def plan_path(self, env):
        planner = AStarPlanner()
        planner.set_environment(
            start=env.agent_pos,
            goal=env.goal_pos,
            grid_size=env.grid_size,
            obstacles=env.obstacles
        )
        return planner.plan()

    def is_valid_path(self, path):
        return path and len(path) > 1

    def perturb_environment(self, env):
        strategies = ['remove', 'move', 'mutate']
        self.rng.shuffle(strategies)

        for strategy in strategies:
            new_env = env.clone()

            if strategy == 'remove':
                if new_env.obstacle_shapes:
                    obs_id = self.rng.choice(list(new_env.obstacle_shapes.keys()))
                    del new_env.obstacle_shapes[obs_id]
                    new_env.update_obstacles_from_shapes()

            elif strategy == 'move':
                if new_env.obstacle_shapes:
                    obs_id = self.rng.choice(list(new_env.obstacle_shapes.keys()))
                    shape = new_env.obstacle_shapes[obs_id]
                    dx, dy = self.rng.choice([(-1,0), (1,0), (0,-1), (0,1)])
                    new_shape = [
                        (max(0, min(self.grid_size - 1, x + dx)),
                         max(0, min(self.grid_size - 1, y + dy)))
                        for x, y in shape
                    ]
                    new_env.obstacle_shapes[obs_id] = new_shape
                    new_env.update_obstacles_from_shapes()

            elif strategy == 'mutate':
                if len(new_env.obstacle_shapes) >= 2:
                    ids = self.rng.sample(list(new_env.obstacle_shapes.keys()), 2)
                    grow_id, shrink_id = ids
                    grow_shape = new_env.obstacle_shapes[grow_id]
                    shrink_shape = new_env.obstacle_shapes[shrink_id]
                    if shrink_shape:
                        moved = shrink_shape.pop()
                        grow_shape.append(moved)
                        new_env.update_obstacles_from_shapes()

            new_path = self.plan_path(new_env)
            if self.is_valid_path(new_path):
                return new_env, new_path

        return None, None

    def save_pair(self, i, env_a, path_a, env_b, path_b):
        pair_dir = os.path.join("contrastive_envs", f"pair_{i:03d}")
        os.makedirs(pair_dir, exist_ok=True)
        output = {
            "original_environment": {
                "grid_size": env_a.grid_size,
                "num_obstacles": env_a.num_obstacles,
                "agent_pos": env_a.agent_pos,
                "goal_pos": env_a.goal_pos,
                "obstacle_shapes": env_a.obstacle_shapes,
                "path": path_a
            },
            "contrastive_environment": {
                "grid_size": env_b.grid_size,
                "num_obstacles": env_b.num_obstacles,
                "agent_pos": env_b.agent_pos,
                "goal_pos": env_b.goal_pos,
                "obstacle_shapes": env_b.obstacle_shapes,
                "path": path_b
            }
        }
        with open(os.path.join(pair_dir, "pair.json"), "w") as f:
            json.dump(output, f, indent=2)

    def generate(self, count=100, attempts_per_env=50):
        success = 0
        seed_for_env = 0 #self.rng.randint(0, 999999)
        for i in range(count):
            for _ in range(attempts_per_env):
                seed_for_env += 1
                env = GridWorldEnv(grid_size=self.grid_size, num_obstacles=self.num_obstacles, seed=seed_for_env)
                env.agent_pos = env.generate_random_position()
                env.goal_pos = env.generate_random_position()
                path_a = self.plan_path(env)

                if not self.is_valid_path(path_a):
                    continue

                perturbed_env, path_b = self.perturb_environment(env)
                if not perturbed_env or path_b == path_a:
                    continue

                self.save_pair(success, env, path_a, perturbed_env, path_b)
                print(f"[✓] Pair {success+1} saved.")
                success += 1
                break
            else:
                print(f"[x] Failed to generate pair {i+1}")

        print(f"\nFinished. Total contrastive pairs generated: {success}/{count}")


if __name__ == "__main__":
    gen = ContrastiveEnvironmentGenerator(grid_size=12, num_obstacles=10)
    gen.generate(count=10000)
