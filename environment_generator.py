import random
import os
import json
from grid_world_env import GridWorldEnv
from path_planners.astar import AStarPlanner

class EnvironmentGenerator:
    """
    Generates grid world environments that are either feasible or infeasible.
    A feasible environment has a valid path from start to goal.
    An infeasible environment has no valid path from start to goal.
    """
    def __init__(self, grid_size=10, num_obstacles=8, seed=None):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
    def is_feasible(self, env, planner=None):
        """Check if the environment has a valid path from start to goal."""
        if planner is None:
            planner = AStarPlanner()
        planner.set_environment(
            start=env.agent_pos,
            goal=env.goal_pos,
            grid_size=env.grid_size,
            obstacles=env.obstacles
        )
        result = planner.plan(return_steps=False)  # Specify return_steps=False
        # Handle both cases - either a single path or a tuple with path as first element
        path = result[0] if isinstance(result, tuple) else result
        return path is not None and len(path) > 0
        
    def generate_position(self, env):
        """Generate a random position that is not an obstacle."""
        while True:
            row = random.randint(0, env.grid_size - 1)
            col = random.randint(0, env.grid_size - 1)
            pos = [row, col]
            if pos not in env.obstacles and (env.agent_pos is None or pos != env.agent_pos):
                return pos
                
    def generate_environment(self, feasible=True, max_attempts=100, start=None, goal=None, infeasibility_mode=None, planner_class=None, env_seed=None):
        """
        Generate an environment that meets the specified feasibility criteria.
        
        Args:
            feasible (bool): Whether to generate a feasible environment
            max_attempts (int): Maximum number of attempts
            env_seed (int): Seed for this specific environment generation
            
        Returns:
            env (GridWorldEnv): The generated environment, or None if failed
        """
        # Set seed for this environment if provided
        if env_seed is not None:
            random.seed(env_seed)
            
        for attempt in range(max_attempts):
            # Create a new environment with deterministic seed
            env_creation_seed = env_seed + attempt if env_seed is not None else None
            env = GridWorldEnv(grid_size=self.grid_size, num_obstacles=self.num_obstacles, seed=env_creation_seed)
            
            # Set random start and goal positions
            if start:
                env.agent_pos = start
            else:
                env.agent_pos = self.generate_position(env)

            if goal:
                env.goal_pos = goal
            else:
                env.goal_pos = self.generate_position(env)

            if not feasible and infeasibility_mode == "block_path":
                if planner_class is None:
                    planner_class = AStarPlanner
                self._block_planner_path(env, planner_class)

            # Check if the environment meets our criteria
            env_feasible = self.is_feasible(env)
            
            # Return if it matches what we want
            if env_feasible == feasible:
                return env
            
            # Print progress for longer runs
            if attempt > 0 and attempt % 25 == 0:
                print(f"  Attempt {attempt}/{max_attempts} - no matching environment yet")
                
        print(f"Failed to generate environment after {max_attempts} attempts")
        return None  # Failed to generate a suitable environment
    
    def _block_planner_path(self, env, planner_class, max_block_attempts=5):
        """
        Iteratively blocks the path returned by the planner until the environment becomes infeasible.

        Args:
            env (GridWorldEnv): The environment instance
            planner_class (class): The planner class to use (must have set_environment() and plan())
            max_block_attempts (int): Max number of blocking iterations
        """
        # Store the original number of obstacles to maintain the limit
        original_num_obstacles = env.num_obstacles
        
        for attempt in range(max_block_attempts):
            # Set up planner
            planner = planner_class()
            planner.set_environment(
                start=env.agent_pos,
                goal=env.goal_pos,
                grid_size=env.grid_size,
                obstacles=env.obstacles
            )

            result = planner.plan(return_steps=False)
            path = result[0] if isinstance(result, tuple) else result

            if not path or len(path) == 0:
                print(f"Path already blocked after {attempt} blocking attempts")
                break  # Already infeasible

            # Instead of adding new obstacles, modify existing ones or use a different approach
            # Option 1: Extend existing obstacle shapes instead of creating new ones
            blocking_cells = [cell for cell in path[1:-1] if cell not in env.obstacles]
            
            if blocking_cells:
                # Try to extend existing obstacle shapes with the blocking cells
                if env.obstacle_shapes:
                    # Find the obstacle shape with the most cells to extend
                    largest_shape_id = max(env.obstacle_shapes.keys(), 
                                        key=lambda k: len(env.obstacle_shapes[k]))
                    
                    # Add blocking cells to the largest existing obstacle shape
                    env.obstacle_shapes[largest_shape_id].extend(blocking_cells)
                    env.obstacles.extend(blocking_cells)
                    
                    print(f"Extended obstacle {largest_shape_id} with {len(blocking_cells)} blocking cells")
                else:
                    # If no existing obstacles, create one (shouldn't happen in normal cases)
                    env.obstacle_shapes[0] = blocking_cells
                    env.obstacles.extend(blocking_cells)
            else:
                print(f"No new cells to block on attempt {attempt}.")
                break
        
    
    def generate_environments_batch(self, n, feasible=True, max_attempts_per_env=100, max_total_attempts=10000, infeasibility_mode=None, planner_class=None, start_seed=None):
        """
        Generate n environments that meet the feasibility criteria.
        
        Args:
            n (int): Number of environments to generate
            feasible (bool): Whether to generate feasible environments
            max_attempts_per_env (int): Maximum attempts per environment
            max_total_attempts (int): Maximum total attempts across all environments
            start_seed (int): Starting seed for deterministic generation
            
        Returns:
            list: Generated environments
            int: Number of environments successfully generated
        """
        environments = []
        total_attempts = 0
        
        for i in range(n):
            attempts_left = min(max_attempts_per_env, max_total_attempts - total_attempts)
            
            if attempts_left <= 0:
                print(f"Reached maximum total attempts ({max_total_attempts}). "
                    f"Generated {len(environments)}/{n} environments.")
                break
                
            print(f"Generating environment {i+1}/{n}...")
            
            # Use deterministic seed for each environment
            env_seed = start_seed + i if start_seed is not None else None
            env = self.generate_environment(
                feasible=feasible, 
                max_attempts=attempts_left, 
                infeasibility_mode=infeasibility_mode, 
                planner_class=planner_class,
                env_seed=env_seed
            )
            
            if env:
                environments.append(env)
                print(f"Successfully generated environment {i+1}/{n}")
            else:
                print(f"Failed to generate environment {i+1}/{n} within attempt limits")
                
            # Update total attempts
            total_attempts += attempts_left if not env else 1
            
        return environments, len(environments)
        
    def save_environment(self, env, filename):
        """Save environment to a JSON file."""
        env_data = {
            "grid_size": env.grid_size,
            "num_obstacles": env.num_obstacles,
            "obstacle_shapes": env.obstacle_shapes,
            "agent_pos": env.agent_pos,
            "goal_pos": env.goal_pos
        }
        
        with open(filename, 'w') as f:
            json.dump(env_data, f, indent=2)


def main():
    """
    Main function to generate environments and save them to folders.
    Generates 10000 infeasible and 10000 feasible environments with many obstacles.
    """
    # Set master seed for reproducible generation
    seed = 42
    random.seed(seed)
    
    # Parameters for complex environments with many obstacles
    n = 10000  # Number of environments to generate
    grid_size = 15  # Larger grid to accommodate more obstacles
    num_obstacles = 15  # Many obstacles for complex explanations
    
    # Create output directories
    base_dir = "environments"
    infeasible_dir = os.path.join(base_dir, "infeasible")
    feasible_dir = os.path.join(base_dir, "feasible")
    
    os.makedirs(infeasible_dir, exist_ok=True)
    os.makedirs(feasible_dir, exist_ok=True)
    
    # Initialize generator with seed
    generator = EnvironmentGenerator(grid_size=grid_size, num_obstacles=num_obstacles, seed=seed)
    
    print(f"Generating environments with grid_size={grid_size}, num_obstacles={num_obstacles}")
    print(f"Using seed: {seed}")
    print("=" * 60)
    
    # Generate infeasible environments with deterministic seeds
    print("Generating 10000 infeasible environments...")
    infeasible_envs, infeasible_count = generator.generate_environments_batch(
        n=n,
        feasible=False,
        max_attempts_per_env=200,
        max_total_attempts=50000,
        infeasibility_mode="block_path",
        planner_class=AStarPlanner,
        start_seed=seed 
    )
    
    # Save infeasible environments
    print(f"\nSaving {infeasible_count} infeasible environments...")
    for i, env in enumerate(infeasible_envs):
        filename = os.path.join(infeasible_dir, f"infeasible_env_{i+1:05d}.json")
        generator.save_environment(env, filename)
        if (i + 1) % 1000 == 0:
            print(f"Saved {i+1}/{infeasible_count} infeasible environments")
    
    print(f"Completed saving {infeasible_count} infeasible environments to {infeasible_dir}")
    print("=" * 60)
    
    # Generate feasible environments with deterministic seeds
    print("Generating 10000 feasible environments...")
    feasible_envs, feasible_count = generator.generate_environments_batch(
        n=n,
        feasible=True,
        max_attempts_per_env=200,
        max_total_attempts=50000,
        start_seed=seed
    )
    
    # Save feasible environments
    print(f"\nSaving {feasible_count} feasible environments...")
    for i, env in enumerate(feasible_envs):
        filename = os.path.join(feasible_dir, f"feasible_env_{i+1:05d}.json")
        generator.save_environment(env, filename)
        if (i + 1) % 1000 == 0:
            print(f"Saved {i+1}/{feasible_count} feasible environments")
    
    print(f"Completed saving {feasible_count} feasible environments to {feasible_dir}")
    print("=" * 60)
    
    # Summary
    print("GENERATION SUMMARY:")
    print(f"Seed used: {seed}")
    print(f"Infeasible environments: {infeasible_count}/10000")
    print(f"Feasible environments: {feasible_count}/10000")
    print(f"Total environments generated: {infeasible_count + feasible_count}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Obstacles per environment: {num_obstacles}")
    print(f"Output directories:")
    print(f"  - Infeasible: {infeasible_dir}")
    print(f"  - Feasible: {feasible_dir}")

def main_loop():
    grid_sizes = list(range(10, 16))
    obstacle_numbers = list(range(5, 16))

    for grid_size in grid_sizes:
        for num_obstacles in obstacle_numbers:
            print(f"Generating environments with grid_size={grid_size}, num_obstacles={num_obstacles}")
            
            # Set master seed for reproducible generation
            seed = 42
            random.seed(seed)
            
            # Parameters for complex environments with many obstacles
            n = 10000  # Number of environments to generate
            
            # Create output directories
            base_dir = os.path.join("environments/", f"grid_{grid_size}_obstacles_{num_obstacles}")
            infeasible_dir = os.path.join(base_dir, "infeasible")
            feasible_dir = os.path.join(base_dir, "feasible")
            
            os.makedirs(infeasible_dir, exist_ok=True)
            os.makedirs(feasible_dir, exist_ok=True)
            
            # Initialize generator with seed
            generator = EnvironmentGenerator(grid_size=grid_size, num_obstacles=num_obstacles, seed=seed)
            
            print(f"Generating environments with grid_size={grid_size}, num_obstacles={num_obstacles}")
            print(f"Using seed: {seed}")
            print("=" * 60)
            
            # Generate infeasible environments with deterministic seeds
            print("Generating 10000 infeasible environments...")
            infeasible_envs, infeasible_count = generator.generate_environments_batch(
                n=n,
                feasible=False,
                max_attempts_per_env=200,
                max_total_attempts=50000,
                infeasibility_mode="block_path",
                planner_class=AStarPlanner,
                start_seed=seed 
            )
            
            # Save infeasible environments
            print(f"\nSaving {infeasible_count} infeasible environments...")
            for i, env in enumerate(infeasible_envs):
                filename = os.path.join(infeasible_dir, f"infeasible_env_{i+1:05d}.json")
                generator.save_environment(env, filename)
                if (i + 1) % 1000 == 0:
                    print(f"Saved {i+1}/{infeasible_count} infeasible environments")
            
            print(f"Completed saving {infeasible_count} infeasible environments to {infeasible_dir}")
            print("=" * 60)
            
            # Generate feasible environments with deterministic seeds
            print("Generating 10000 feasible environments...")
            feasible_envs, feasible_count = generator.generate_environments_batch(
                n=n,
                feasible=True,
                max_attempts_per_env=200,
                max_total_attempts=50000,
                start_seed=seed
            )
            
            # Save feasible environments
            print(f"\nSaving {feasible_count} feasible environments...")
            for i, env in enumerate(feasible_envs):
                filename = os.path.join(feasible_dir, f"feasible_env_{i+1:05d}.json")
                generator.save_environment(env, filename)
                if (i + 1) % 1000 == 0:
                    print(f"Saved {i+1}/{feasible_count} feasible environments")
            
            print(f"Completed saving {feasible_count} feasible environments to {feasible_dir}")
            print("=" * 60)
            
            # Summary
            print("GENERATION SUMMARY:")
            print(f"Seed used: {seed}")
            print(f"Infeasible environments: {infeasible_count}/10000")
            print(f"Feasible environments: {feasible_count}/10000")
            print(f"Total environments generated: {infeasible_count + feasible_count}")
            print(f"Grid size: {grid_size}x{grid_size}")
            print(f"Obstacles per environment: {num_obstacles}")
            print(f"Output directories:")
            print(f"  - Infeasible: {infeasible_dir}")
            print(f"  - Feasible: {feasible_dir}")

if __name__ == "__main__":
    #main()
    main_loop()