import random
import os
import json
from gui import GridWorldEnv
from path_planning.astar import AStarPlanner

class EnvironmentGenerator:
    """
    Generates grid world environments that are either feasible or infeasible.
    A feasible environment has a valid path from start to goal.
    An infeasible environment has no valid path from start to goal.
    """
    def __init__(self, grid_size=10, num_obstacles=8):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        
    def is_feasible(self, env):
        """Check if the environment has a valid path from start to goal."""
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
                
    def generate_environment(self, feasible=True, max_attempts=100):
        """
        Generate an environment that meets the specified feasibility criteria.
        
        Args:
            feasible (bool): Whether to generate a feasible environment
            max_attempts (int): Maximum number of attempts
            
        Returns:
            env (GridWorldEnv): The generated environment, or None if failed
        """
        for attempt in range(max_attempts):
            # Create a new environment
            env = GridWorldEnv(grid_size=self.grid_size, num_obstacles=self.num_obstacles)
            
            # Set random start and goal positions
            env.agent_pos = self.generate_position(env)
            env.goal_pos = self.generate_position(env)
            
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
    
    def generate_environments_batch(self, n, feasible=True, max_attempts_per_env=100, max_total_attempts=10000):
        """
        Generate n environments that meet the feasibility criteria.
        
        Args:
            n (int): Number of environments to generate
            feasible (bool): Whether to generate feasible environments
            max_attempts_per_env (int): Maximum attempts per environment
            max_total_attempts (int): Maximum total attempts across all environments
            
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
            env = self.generate_environment(feasible=feasible, max_attempts=attempts_left)
            
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