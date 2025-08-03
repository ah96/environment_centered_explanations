import argparse
import os
import shutil
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from batch_experiment import BatchExperimentRunner
from gui import GridWorldEnv
from multiprocessing import Pool, cpu_count
from functools import partial

def get_ranked_obstacles_from_explanation(explainer, explainer_name, env):
    """
    Get a ranked list of all obstacles based on explanation importance
    
    Args:
        explainer: Explainer object
        explainer_name: Name of the explainer
        env: Environment object
    
    Returns:
        List of obstacle IDs ranked by importance (most important first)
    """
    try:
        obstacle_ids = list(env.obstacle_shapes.keys())
        
        if explainer_name in ["LIME", "SHAP"]:
            explanation = explainer.explain(num_samples=50)
            
            if isinstance(explanation, dict):
                # If explanation is a dictionary mapping obstacle_id -> value
                importance_values = [(obs_id, abs(val)) for obs_id, val in explanation.items() if obs_id in obstacle_ids]
                importance_values.sort(key=lambda x: x[1], reverse=True)
                ranked_obstacles = [obs_id for obs_id, _ in importance_values]
                
                # Add any missing obstacles at the end
                missing_obstacles = [obs_id for obs_id in obstacle_ids if obs_id not in ranked_obstacles]
                ranked_obstacles.extend(missing_obstacles)
                
            elif isinstance(explanation, (list, tuple, np.ndarray)) and len(explanation) > 0:
                # Handle as list/array - map indices to obstacle IDs
                importance_values = [(i, abs(val)) for i, val in enumerate(explanation) if i < len(obstacle_ids)]
                importance_values.sort(key=lambda x: x[1], reverse=True)
                ranked_obstacles = [obstacle_ids[i] for i, _ in importance_values]
                
                # Add any missing obstacles at the end
                missing_obstacles = [obstacle_ids[i] for i in range(len(obstacle_ids)) if i >= len(explanation)]
                ranked_obstacles.extend(missing_obstacles)
            else:
                # Fallback to original order
                ranked_obstacles = obstacle_ids
                
        else:  # Anchors
            anchors = explainer.explain(num_samples=50)
            anchor_obstacles = []
            
            if isinstance(anchors, dict) and "anchors" in anchors and anchors["anchors"]:
                anchor_obstacles = [obs for obs in anchors["anchors"] if obs in obstacle_ids]
            elif isinstance(anchors, (list, tuple)):
                anchor_obstacles = [obs for obs in anchors if obs in obstacle_ids]
            
            # Put anchor obstacles first, then the rest
            ranked_obstacles = anchor_obstacles + [obs for obs in obstacle_ids if obs not in anchor_obstacles]
        
        return ranked_obstacles
        
    except Exception as e:
        print(f"Warning: Error ranking obstacles for {explainer_name}: {e}")
        return list(env.obstacle_shapes.keys())

def calculate_path_optimality(path, optimal_path):
    """
    Calculate how optimal a path is compared to the theoretical optimal path
    
    Args:
        path: The current path
        optimal_path: The optimal path (with no obstacles)
        
    Returns:
        Optimality score (1.0 means it's optimal, lower means less optimal)
    """
    if not path or not optimal_path:
        return 0.0
        
    # Optimality is ratio of optimal path length to current path length
    # (since shorter paths are better)
    return len(optimal_path) / len(path)

def calculate_obstacle_removal_efficiency(removed_obstacles, total_obstacles):
    """
    Calculate efficiency of obstacle removal (fewer removals is better)
    
    Args:
        removed_obstacles: Number of obstacles removed
        total_obstacles: Total number of obstacles
        
    Returns:
        Efficiency score (1.0 is best - no obstacles removed, 0.0 is worst - all removed)
    """
    if total_obstacles == 0:
        return 1.0  # No obstacles to remove
        
    return 1.0 - (removed_obstacles / total_obstacles)

def evaluate_path_after_obstacle_removal(env, planner, explainer, explainer_name):
    """
    Evaluate how many obstacles need to be removed before a path is found,
    and measure the optimality of that path
    
    Args:
        env: Environment object
        planner: Planner class
        explainer: Explainer object
        explainer_name: Name of the explainer
        
    Returns:
        Dictionary with metrics (obstacles_removed, path_length, etc.)
    """
    # Get ranked obstacles from explanation
    ranked_obstacles = get_ranked_obstacles_from_explanation(explainer, explainer_name, env)
    
    # Calculate optimal path (if no obstacles existed)
    obstacle_free_env = env.clone()
    obstacle_free_env.obstacles = []
    obstacle_free_env.obstacle_shapes = {}
    
    optimal_planner = planner()
    optimal_planner.set_environment(
        start=obstacle_free_env.agent_pos,
        goal=obstacle_free_env.goal_pos,
        grid_size=obstacle_free_env.grid_size,
        obstacles=obstacle_free_env.obstacles
    )
    
    result = optimal_planner.plan(return_steps=False)
    optimal_path = result[0] if isinstance(result, tuple) else result
    
    # Start removing obstacles one by one
    modified_env = env.clone()
    total_obstacles = len(ranked_obstacles)
    
    for k, obstacle_id in enumerate(ranked_obstacles, 1):
        # Remove obstacle
        if obstacle_id in modified_env.obstacle_shapes:
            points_to_remove = modified_env.obstacle_shapes[obstacle_id]
            modified_env.obstacles = [p for p in modified_env.obstacles if p not in points_to_remove]
            del modified_env.obstacle_shapes[obstacle_id]
        
        # Check if path exists now
        planner_instance = planner()
        planner_instance.set_environment(
            start=modified_env.agent_pos,
            goal=modified_env.goal_pos,
            grid_size=modified_env.grid_size,
            obstacles=modified_env.obstacles
        )
        
        result = planner_instance.plan(return_steps=False)
        path = result[0] if isinstance(result, tuple) else result
        
        if path and len(path) > 0:
            # Path found after removing k obstacles
            path_length = len(path)
            optimal_length = len(optimal_path) if optimal_path else 0
            
            # Calculate metrics
            obstacle_efficiency = calculate_obstacle_removal_efficiency(k, total_obstacles)
            path_optimality = calculate_path_optimality(path, optimal_path) if optimal_path else 0.0
            
            return {
                "obstacles_removed": k,
                "obstacles_removed_percentage": k / total_obstacles if total_obstacles > 0 else 0,
                "obstacle_removal_efficiency": obstacle_efficiency,
                "path_length": path_length,
                "optimal_path_length": optimal_length,
                "path_optimality": path_optimality,
                "success": True
            }
    
    # No path found even after removing all obstacles
    return {
        "obstacles_removed": total_obstacles,
        "obstacles_removed_percentage": 1.0,
        "obstacle_removal_efficiency": 0.0,
        "path_length": 0,
        "optimal_path_length": len(optimal_path) if optimal_path else 0,
        "path_optimality": 0.0,
        "success": False
    }

def run_single_env_experiment(env_path, planners, explanations):
    from batch_experiment import BatchExperimentRunner
    from gui import GridWorldEnv
    import time

    runner = BatchExperimentRunner()
    results = []
    env_name = os.path.basename(env_path)
    env = runner.load_environment(env_path)

    for planner_name in planners:
        planner_class = runner.planners[planner_name]
        planner_instance = planner_class()
        planner_instance.set_environment(
            start=env.agent_pos,
            goal=env.goal_pos,
            grid_size=env.grid_size,
            obstacles=env.obstacles
        )

        for explainer_name in explanations:
            explainer_class = runner.explainers[explainer_name]
            explainer = explainer_class()
            explainer.set_environment(env, planner_instance)

            start_time = time.time()
            metrics = evaluate_path_after_obstacle_removal(env, planner_class, explainer, explainer_name)
            explanation_time = time.time() - start_time

            results.append({
                "Environment": env_name,
                "Explanation": explainer_name,
                "Planner": planner_name,
                "Obstacles_Removed": metrics["obstacles_removed"],
                "Obstacles_Removed_Percentage": metrics["obstacles_removed_percentage"],
                "Obstacle_Removal_Efficiency": metrics["obstacle_removal_efficiency"],
                "Path_Length": metrics["path_length"],
                "Optimal_Path_Length": metrics["optimal_path_length"],
                "Path_Optimality": metrics["path_optimality"],
                "Success": metrics["success"],
                "Explanation_Time_s": explanation_time
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiments to evaluate when plans become possible")
    parser.add_argument("--env_dir", type=str, default="environments/infeasible")
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--output", type=str, default="when_do_you_not_fail_results.csv")
    parser.add_argument("--average_only", default=True, action="store_true")
    args = parser.parse_args()

    def create_clean_folder(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    create_clean_folder("when")

    explanations = ["SHAP", "LIME"]
    planners = ["A*", "Dijkstra", "Theta*", "BFS", "DFS", "Greedy Best-First"]
    NUM_ENVS = args.num_envs

    for env_size in range(10, 21, 2):
        for obstacle_num in range(5, 21, 2):
            OUTPUT = f"when/when_do_you_not_fail_results_grid_{env_size}_obstacles_{obstacle_num}.csv"
            ENV_DIR = f"environments/grid_{env_size}_obstacles_{obstacle_num}/infeasible"
            env_files = [f for f in os.listdir(ENV_DIR) if f.endswith('.json') and 'infeasible' in f]
            if not env_files:
                print(f"No infeasible environments found in {ENV_DIR}")
                continue

            env_files = env_files[:NUM_ENVS]
            env_paths = [os.path.join(ENV_DIR, f) for f in env_files]

            print(f"Loaded {len(env_paths)} infeasible environments")

            num_workers = min(20, cpu_count() // 2)
            run_func = partial(run_single_env_experiment, planners=planners, explanations=explanations)

            print(f"Running with {num_workers} parallel workers...")

            with Pool(processes=num_workers) as pool:
                all_results_nested = list(tqdm(pool.imap_unordered(run_func, env_paths), total=len(env_paths), desc="Processing"))
            
            results_data = [item for sublist in all_results_nested for item in sublist]
            df = pd.DataFrame(results_data)

            if args.average_only:
                avg_df = df.groupby([
                    "Explanation", "Planner"
                ]).agg({
                    "Obstacles_Removed": "mean",
                    "Obstacles_Removed_Percentage": "mean",
                    "Obstacle_Removal_Efficiency": "mean",
                    "Path_Length": "mean",
                    "Optimal_Path_Length": "mean",
                    "Path_Optimality": "mean",
                    "Success": "mean",
                    "Explanation_Time_s": "mean"
                }).reset_index()

                if os.path.exists(OUTPUT):
                    existing_df = pd.read_csv(OUTPUT)
                    combined_df = pd.concat([existing_df, avg_df], ignore_index=True)
                    combined_df.to_csv(OUTPUT, index=False)
                else:
                    avg_df.to_csv(OUTPUT, index=False)

                print(f"Averaged rows: {len(avg_df)}")
                print("\nSummary Statistics (Averaged):")
                print(avg_df.describe())
            else:
                if os.path.exists(OUTPUT):
                    existing_df = pd.read_csv(OUTPUT)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv(OUTPUT, index=False)
                else:
                    df.to_csv(OUTPUT, index=False)

                print(f"Total rows: {len(df)}")
                print("\nSummary Statistics:")
                print(df.describe())

if __name__ == "__main__":
    main()