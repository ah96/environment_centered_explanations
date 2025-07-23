import os
import csv
import time
import json
from datetime import datetime

# Import path planning algorithms
from path_planners.astar import AStarPlanner
from path_planners.dijkstra import DijkstraPlanner
from path_planners.theta_star import ThetaStarPlanner
from path_planners.bfs import BFSPlanner
from path_planners.greedy_best_first import GreedyBestFirstPlanner
from path_planners.dfs import DFSPlanner
from path_planners.rrt import RRTPlanner
from path_planners.rrt_star import RRTStarPlanner
from path_planners.prm import PRMPlanner

# Import explanation methods
from explainers.lime_explainer import LimeExplainer
from explainers.anchors_explainer import AnchorsExplainer
from explainers.shap_explainer import SHAPExplainer

from environment_generator import EnvironmentGenerator
from gui import GridWorldEnv
from metrics import compute_path_metrics

class BatchExperimentRunner:
    def __init__(self):
        # Define available algorithms and explainers
        self.planners = {
            "A*": AStarPlanner,
            "Dijkstra": DijkstraPlanner,
            "Theta*": ThetaStarPlanner,
            "BFS": BFSPlanner,
            "DFS": DFSPlanner,
            "Greedy Best-First": GreedyBestFirstPlanner,
            "RRT": RRTPlanner,
            "RRT*": RRTStarPlanner,
            "PRM": PRMPlanner
        }
        
        self.explainers = {
            "LIME": LimeExplainer,
            "Anchors": AnchorsExplainer,
            "SHAP": SHAPExplainer
        }
        
    def load_environment(self, filepath):
        """Load environment from a JSON file."""
        with open(filepath, 'r') as f:
            env_data = json.load(f)
            
        # Create new environment with loaded parameters
        env = GridWorldEnv(grid_size=env_data["grid_size"], 
                           num_obstacles=env_data.get("num_obstacles", 0))
        
        # Load obstacle shapes
        if "obstacle_shapes" in env_data:
            # Convert string keys back to integers
            obstacle_shapes = {int(k): v for k, v in env_data["obstacle_shapes"].items()}
            env.obstacle_shapes = obstacle_shapes
            
            # Reconstruct flat obstacles list
            env.obstacles = []
            for shape_points in env.obstacle_shapes.values():
                env.obstacles.extend(shape_points)
        
        # Load agent and goal positions
        env.agent_pos = env_data.get("agent_pos")
        env.goal_pos = env_data.get("goal_pos")
        
        return env
    
    
    def generate_environments(self, n, feasible=True, grid_size=10, num_obstacles=8, 
                         max_attempts_per_env=100, max_total_attempts=10000, infeasibility_mode=None, planner_name="A*"):
        """
        Generate n environments and save them to the environments folder.
        
        Args:
            n (int): Number of environments to generate
            feasible (bool): Whether to generate feasible environments
            grid_size (int): Size of the grid
            num_obstacles (int): Number of obstacles
            max_attempts_per_env (int): Maximum attempts per environment
            max_total_attempts (int): Maximum total attempts across all environments
            
        Returns:
            list: Paths to the generated environment files
        """
        # Create environments directory if it doesn't exist
        os.makedirs("environments", exist_ok=True)
        
        # Initialize generator
        generator = EnvironmentGenerator(grid_size=grid_size, num_obstacles=num_obstacles)

        planner_class = self.planners.get(planner_name)
        
        # Generate environments in batch
        environments, count = generator.generate_environments_batch(
            n, 
            feasible=feasible,
            max_attempts_per_env=max_attempts_per_env,
            max_total_attempts=max_total_attempts,
            infeasibility_mode=infeasibility_mode,
            planner_class=planner_class
        )
        
        # Save the environments
        env_paths = []
        for i, env in enumerate(environments):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            feasibility = "feasible" if feasible else "infeasible"
            filename = f"environments/env_{feasibility}_{i}_{timestamp}.json"
            generator.save_environment(env, filename)
            env_paths.append(filename)
        
        print(f"Saved {count} environments to the environments directory")
        return env_paths
        
    def run_experiment(self, env_paths, planner_name, explainer_name, results_csv):
        """
        Run experiment on the given environments.
        
        Args:
            env_paths (list): Paths to environment files
            planner_name (str): Name of the planner to use
            explainer_name (str): Name of the explainer to use
            results_csv (str): Path to CSV file to save results
        """
        # Check if planner and explainer exist
        if planner_name not in self.planners:
            raise ValueError(f"Unknown planner: {planner_name}")
        if explainer_name not in self.explainers:
            raise ValueError(f"Unknown explainer: {explainer_name}")
            
        # Create CSV file with headers
        with open(results_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Environment", 
                "Planner", 
                "Explainer",
                "Original Path Length",
                "Original Execution Time",
                "Obstacles Removed",
                "New Path Length",
                "New Execution Time",
                "Path Length Difference",
                "Path Length Change %",
                "Execution Time Difference",
                "Path Restored",
                "Path Success"
            ])
            
        # Process each environment
        for env_path in env_paths:
            # Load environment
            env = self.load_environment(env_path)
            
            # Create planner
            planner_class = self.planners[planner_name]
            planner = planner_class()
            planner.set_environment(
                start=env.agent_pos,
                goal=env.goal_pos,
                grid_size=env.grid_size,
                obstacles=env.obstacles
            )
            
            # Run original path planning
            start_time = time.time()
            result = planner.plan(return_steps=False)
            original_path = result[0] if isinstance(result, tuple) else result
            original_exec_time = time.time() - start_time
            
            # Proceed only if there's a path or we're analyzing infeasible environments
            original_path_length = len(original_path) if original_path else 0
            
            # Create explainer
            explainer_class = self.explainers[explainer_name]
            explainer = explainer_class()
            explainer.set_environment(env, planner)
            
            # Get explanation (depends on explainer type)
            obstacles_removed = []
            if explainer_name == "LIME":
                explanation = explainer.explain(num_samples=30)
                # Get the most important obstacle to remove (highest absolute value)
                if isinstance(explanation, (list, tuple)) and len(explanation) > 0:
                    importance_values = [(i, abs(val)) for i, val in enumerate(explanation)]
                    importance_values.sort(key=lambda x: x[1], reverse=True)
                    if importance_values:
                        obstacle_to_remove = list(env.obstacle_shapes.keys())[importance_values[0][0]]
                        obstacles_removed = [obstacle_to_remove]
                # If explanation is a numpy array
                elif hasattr(explanation, 'size') and explanation.size > 0:
                    importance_values = [(i, abs(val)) for i, val in enumerate(explanation)]
                    importance_values.sort(key=lambda x: x[1], reverse=True)
                    if importance_values:
                        obstacle_to_remove = list(env.obstacle_shapes.keys())[importance_values[0][0]]
                        obstacles_removed = [obstacle_to_remove]
            
            elif explainer_name == "Anchors":
                anchors = explainer.explain(num_samples=30)
                if anchors and "anchors" in anchors and anchors["anchors"]:
                    obstacles_removed = [anchors["anchors"][0]]
            
            elif explainer_name == "SHAP":
                shap_values = explainer.explain(num_samples=30)
                if shap_values and len(shap_values) > 0:
                    importance_values = [(i, abs(val)) for i, val in enumerate(shap_values)]
                    importance_values.sort(key=lambda x: x[1], reverse=True)
                    if importance_values:
                        obstacle_to_remove = list(env.obstacle_shapes.keys())[importance_values[0][0]]
                        obstacles_removed = [obstacle_to_remove]
            
            elif explainer_name == "Counterfactual":
                counterfactuals = explainer.explain(max_subset_size=1)
                if counterfactuals and counterfactuals.get("counterfactuals"):
                    obstacles_removed = counterfactuals["counterfactuals"][0]

            elif explainer_name == "GoalCounterfactual":
                counterfactuals = explainer.explain(goal_condition=True)
                if counterfactuals and counterfactuals.get("counterfactuals"):
                    obstacles_removed = counterfactuals["counterfactuals"][0]
           
            # Remove the obstacles
            original_obstacles = env.obstacles.copy()
            for obs_id in obstacles_removed:
                if obs_id in env.obstacle_shapes:
                    points_to_remove = env.obstacle_shapes[obs_id]
                    env.obstacles = [p for p in env.obstacles if p not in points_to_remove]
            
            # Run path planning again after removing obstacles
            planner = planner_class()  # Create new planner instance
            planner.set_environment(
                start=env.agent_pos,
                goal=env.goal_pos,
                grid_size=env.grid_size,
                obstacles=env.obstacles
            )
            
            start_time = time.time()
            result = planner.plan(return_steps=False)
            new_path = result[0] if isinstance(result, tuple) else result
            new_exec_time = time.time() - start_time
            
            # Compute metrics
            metrics = compute_path_metrics(original_path, new_path, original_exec_time, new_exec_time, env.grid_size)

            # Write results to CSV
            with open(results_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                os.path.basename(env_path),
                planner_name,
                explainer_name,
                metrics["original_length"],
                original_exec_time,
                obstacles_removed,
                metrics["new_length"],
                new_exec_time,
                metrics["path_length_diff"],
                metrics["path_length_change_pct"],
                metrics["exec_time_diff"],
                metrics["path_restored"],
                metrics["path_success"]
            ])
            
            # Restore environment to original state
            env.obstacles = original_obstacles.copy()