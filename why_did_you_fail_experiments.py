import argparse
import os
import pandas as pd
import time
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from batch_experiment import BatchExperimentRunner
from metrics import compute_path_metrics
from environment_generator import EnvironmentGenerator
from gui import GridWorldEnv

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 1.0  # Both empty sets are considered identical
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_explanation_stability(original_features, perturbed_features):
    """
    Calculate explanation stability using Jaccard similarity
    
    Args:
        original_features: Set of important features from original environment
        perturbed_features: Set of important features from perturbed environment
    
    Returns:
        Jaccard similarity score
    """
    return jaccard_similarity(original_features, perturbed_features)

def calculate_faithfulness_score(env, planner, explainer, ranked_obstacles):
    """
    Calculate faithfulness score (fraction of obstacles to flip)
    
    Args:
        env: Original environment object
        planner: Planner object
        explainer: Explainer object
        ranked_obstacles: List of obstacles ranked by importance
    
    Returns:
        Faithfulness score (lower is better)
    """
    # Create a modifiable copy of the environment
    modified_env = env.clone()
    
    total_obstacles = len(ranked_obstacles)
    if total_obstacles == 0:
        return 0  # No obstacles to remove
    
    # Iterate through obstacles and remove them until path is found
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
            return k / total_obstacles
    
    # No path found even after removing all obstacles
    return 1.0

def perturb_environment(env, perturbation_type, num_perturbations):
    """
    Create a perturbed version of the environment based on perturbation type
    
    Args:
        env: Original environment
        perturbation_type: Type of perturbation ('remove', 'move', 'random')
        num_perturbations: Number of perturbations to apply
    
    Returns:
        Perturbed environment
    """
    # Create a copy of the environment
    perturbed_env = GridWorldEnv(grid_size=env.grid_size, num_obstacles=len(env.obstacle_shapes))
    perturbed_env.agent_pos = env.agent_pos
    perturbed_env.goal_pos = env.goal_pos
    perturbed_env.obstacles = env.obstacles.copy()
    perturbed_env.obstacle_shapes = {k: v.copy() for k, v in env.obstacle_shapes.items()}
    
    # Get list of obstacle IDs
    obstacle_ids = list(env.obstacle_shapes.keys())
    
    if not obstacle_ids:
        return perturbed_env
    
    # Apply perturbations based on type
    if perturbation_type == "remove":
        # Randomly remove obstacles (up to num_perturbations)
        num_to_remove = min(num_perturbations, len(obstacle_ids))
        if num_to_remove > 0:
            obstacles_to_remove = np.random.choice(obstacle_ids, size=num_to_remove, replace=False)
            for obs_id in obstacles_to_remove:
                perturbed_env.remove_obstacle_shape(obs_id)
    
    elif perturbation_type == "move":
        # Randomly move obstacles (up to num_perturbations)
        num_to_move = min(num_perturbations, len(obstacle_ids))
        if num_to_move > 0:
            obstacles_to_move = np.random.choice(obstacle_ids, size=num_to_move, replace=False)
            for obs_id in obstacles_to_move:
                perturbed_env.move_obstacle_shape(obs_id)
    
    # elif perturbation_type == "random":
    #     # Add random new obstacles using the environment's method
    #     for _ in range(min(num_perturbations, 5)):  # Limit to prevent overcrowding
    #         # Find a free position for new obstacle
    #         free_positions = []
    #         for r in range(perturbed_env.grid_size):
    #             for c in range(perturbed_env.grid_size):
    #                 if [r, c] not in perturbed_env.obstacles and \
    #                    [r, c] != perturbed_env.agent_pos and \
    #                    [r, c] != perturbed_env.goal_pos:
    #                     free_positions.append([r, c])
            
    #         if free_positions:
    #             pos = free_positions[np.random.randint(len(free_positions))]
    #             new_obstacle_id = max(perturbed_env.obstacle_shapes.keys(), default=0) + 1
    #             perturbed_env.obstacle_shapes[new_obstacle_id] = [pos]
    #             perturbed_env.obstacles.append(pos)
    
    return perturbed_env

def get_important_features(explainer, explainer_name, env, planner_instance, top_k=5):
    """
    Get the top k important features (obstacles) from an explanation
    """
    start_time = time.time()
    
    # For infeasible environments, we need to ensure we get features
    # that are causing the infeasibility
    obstacle_ids = list(env.obstacle_shapes.keys())
    
    # If no explanation is available, use a fallback approach
    # for infeasible environments - consider all obstacles as potentially important
    top_features = set()
    
    try:
        # Try to get explanation from the explainer
        if explainer_name == "LIME":
            explanation = explainer.explain(num_samples=100)  # Increased samples
            
            # LIME returns an array of importance values
            if hasattr(explanation, '__len__') and len(explanation) > 0:
                # Pair each obstacle ID with its importance value
                importance_values = [(obstacle_ids[i], abs(val)) for i, val in enumerate(explanation) if i < len(obstacle_ids)]
                # Sort by importance (highest first)
                importance_values.sort(key=lambda x: x[1], reverse=True)
                # Take top k obstacles
                top_features = set([obs_id for obs_id, _ in importance_values[:min(top_k, len(importance_values))]])
            
        elif explainer_name == "SHAP":
            explanation = explainer.explain(num_samples=100)
            
            # SHAP returns a dictionary mapping obstacle IDs to importance values
            if isinstance(explanation, dict) and explanation:
                # Sort by absolute importance value
                sorted_items = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
                # Take top k obstacles
                top_features = set([obs_id for obs_id, _ in sorted_items[:min(top_k, len(sorted_items))]])
            
        elif explainer_name == "Anchors":
            explanation = explainer.explain(num_samples=100)
            
            # Anchors returns a dictionary with anchor rules
            if explanation:
                # Create a list to collect (obstacle_id, importance) pairs
                important_obstacles = []
                
                # Process each anchor rule
                for rule_str, rule_data in explanation.items():
                    precision = rule_data.get("precision", 0)
                    rule = rule_data.get("rule", [])
                    
                    # Extract obstacles from this rule
                    for i, val in enumerate(rule):
                        if val is not None and i < len(obstacle_ids):
                            # val=1 means keep obstacle, val=0 means remove
                            # If removing (val=0) contributes to path change, it's important
                            importance = precision if val == 0 else precision * 0.5
                            important_obstacles.append((obstacle_ids[i], importance))
                
                # Group by obstacle ID and sum up importances
                obstacle_importance = {}
                for obs_id, imp in important_obstacles:
                    obstacle_importance[obs_id] = obstacle_importance.get(obs_id, 0) + imp
                
                # Sort by importance
                sorted_obstacles = sorted(obstacle_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Take top k
                top_features = set([obs_id for obs_id, _ in sorted_obstacles[:min(top_k, len(sorted_obstacles))]])
        
        # If we still have no features after the explanation, use a fallback
        if not top_features and obstacle_ids:
            # Pick random obstacles as important for testing
            random_obstacles = np.random.choice(obstacle_ids, 
                                               size=min(top_k, len(obstacle_ids)), 
                                               replace=False)
            top_features = set(random_obstacles)
            print(f"Using fallback approach for {explainer_name} with {len(top_features)} features")
    
    except Exception as e:
        print(f"Warning: Error in {explainer_name} explanation: {e}")
        # Use fallback if exception occurs
        if obstacle_ids:
            random_obstacles = np.random.choice(obstacle_ids, 
                                               size=min(top_k, len(obstacle_ids)), 
                                               replace=False)
            top_features = set(random_obstacles)
    
    explanation_time = time.time() - start_time
    return top_features, explanation_time

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

def main():
    parser = argparse.ArgumentParser(description="Run comparison experiments for path planning explanations")
    parser.add_argument("--env_dir", type=str, default="environments/infeasible",
                        help="Directory with existing environments")
    parser.add_argument("--num_envs", type=int, default=100,
                        help="Number of environments to load")
    parser.add_argument("--output", type=str, default="why_did_you_fail_results.csv",
                        help="Output CSV file for results")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top features to consider")
    parser.add_argument("--average_only", default=True, action="store_true",
                        help="When set, only save average results across all environments")
    
    args = parser.parse_args()
    
    # Set up experiment parameters
    explanations = ["SHAP", "LIME", "Anchors"]
    planners = ["A*", "Dijkstra", "Theta*", "BFS", "DFS", "Greedy Best-First"]
    perturbation_counts = [10, 50, 100]
    perturbation_types = ["remove", "move"] #, "random"] # random doesn't make sense, remove and move cover all cases
    
    MAX_PERTURBATION_RETRIES = 100  # Maximum attempts to find an infeasible perturbation

    runner = BatchExperimentRunner()
    
    # Initialize results list for pandas DataFrame
    results_data = []
    
    # Load environments
    env_files = [f for f in os.listdir(args.env_dir) if f.endswith('.json') and 'infeasible' in f]
    if not env_files:
        print(f"No infeasible environments found in {args.env_dir}")
        return
    
    # Take the first num_envs environments or all if fewer available
    env_files = env_files[:args.num_envs]
    env_paths = [os.path.join(args.env_dir, f) for f in env_files]
    
    print(f"Loaded {len(env_paths)} infeasible environments")
    
    # For storing planner robustness data
    planner_features = {}
    
    # Calculate total number of iterations for progress tracking
    total_iterations = len(env_paths) * len(planners) * len(explanations) * len(perturbation_types) * len(perturbation_counts)
    
    # Run experiments with a single progress bar
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for env_path in env_paths:
            env_name = os.path.basename(env_path)
            env = runner.load_environment(env_path)
            
            # Store features for each planner for this environment
            planner_features[env_name] = {}
            
            # For each planner
            for planner_name in planners:
                planner_class = runner.planners[planner_name]
                planner_instance = planner_class()
                planner_instance.set_environment(
                    start=env.agent_pos,
                    goal=env.goal_pos,
                    grid_size=env.grid_size,
                    obstacles=env.obstacles
                )
                
                # Get original path (should be None/empty for infeasible environments)
                result = planner_instance.plan(return_steps=False)
                original_path = result[0] if isinstance(result, tuple) else result
                # for infeasible environments path length is 0
                path_length = len(original_path) if original_path else 0
                
                # For each explanation method
                for explainer_name in explanations:
                    explainer_class = runner.explainers[explainer_name]
                    explainer = explainer_class()
                    explainer.set_environment(env, planner_instance)
                    
                    # Get important features for original environment
                    original_features, explanation_time = get_important_features(
                        explainer, explainer_name, env, planner_instance, top_k=args.top_k
                    )
                    
                    # Store features for planner robustness calculation
                    planner_features[env_name][f"{planner_name}_{explainer_name}"] = original_features
                    
                    # Get ranked list of all obstacles for faithfulness calculation
                    ranked_obstacles = get_ranked_obstacles_from_explanation(explainer, explainer_name, env)
                    
                    # Calculate faithfulness score
                    faithfulness_score = calculate_faithfulness_score(env, planner_class, explainer, ranked_obstacles)
                    
                    # For each perturbation type and count
                    for perturbation_type in perturbation_types:
                        for num_perturbations in perturbation_counts:
                            
                            current_perturbed_env = None
                            final_planner_check_for_perturbed = None
                            found_infeasible_perturbation = False

                            for attempt in range(MAX_PERTURBATION_RETRIES):
                                # Create perturbed environment
                                perturbed_env_candidate = perturb_environment(env, perturbation_type, num_perturbations)
                                
                                # Check if perturbed environment is still infeasible
                                planner_check = planner_class()
                                planner_check.set_environment(
                                    start=perturbed_env_candidate.agent_pos,
                                    goal=perturbed_env_candidate.goal_pos,
                                    grid_size=perturbed_env_candidate.grid_size,
                                    obstacles=perturbed_env_candidate.obstacles
                                )
                                result_check = planner_check.plan(return_steps=False)
                                path_check = result_check[0] if isinstance(result_check, tuple) else result_check
                                
                                if not (path_check and len(path_check) > 0): # Path NOT found (still infeasible)
                                    current_perturbed_env = perturbed_env_candidate
                                    final_planner_check_for_perturbed = planner_check
                                    found_infeasible_perturbation = True
                                    break # Exit retry loop, we found a suitable environment
                                # else: environment became feasible, try perturbing again in the next attempt
                            
                            if not found_infeasible_perturbation:
                                print(f"Warning: Max retries ({MAX_PERTURBATION_RETRIES}) reached for {env_name}, {planner_name}, {explainer_name}, {perturbation_type}, {num_perturbations}. "
                                      f"Could not generate an infeasible perturbed environment. Skipping this combination.")
                                pbar.update(1)  # Update progress bar even when skipped
                                continue # Skip to the next num_perturbations or perturbation_type
                            
                            # Now, current_perturbed_env is an infeasible perturbed environment
                            # and final_planner_check_for_perturbed is the planner instance that confirmed it.
                            
                            # Get explanation for perturbed environment
                            explainer_perturbed = explainer_class()
                            explainer_perturbed.set_environment(current_perturbed_env, final_planner_check_for_perturbed)
                            perturbed_features, _ = get_important_features(
                                explainer_perturbed, explainer_name, current_perturbed_env, final_planner_check_for_perturbed, top_k=args.top_k
                            )
                            
                            # Calculate explanation stability (Jaccard similarity)
                            stability = calculate_explanation_stability(original_features, perturbed_features)
                            
                            # Add results to list
                            results_data.append({
                                "Environment": env_name,
                                "Explanation": explainer_name,
                                "Planner": planner_name,
                                "Num_Perturbations": num_perturbations,
                                "Perturbation_Type": perturbation_type,
                                "Explanation_Stability": stability,
                                "Faithfulness_Score": faithfulness_score,
                                "Planner_Robustness": None,  # Will be filled later
                                "Path_Length": path_length,
                                "Explanation_Time_s": explanation_time
                            })
                            
                            # Update the progress bar
                            pbar.update(1)
    
    # Create DataFrame from results
    df = pd.DataFrame(results_data)
    
    # Calculate planner robustness for each environment and explainer
    for env_name in planner_features:
        for explainer_name in explanations:
            # Get features for each planner with this explainer
            planner_feature_sets = []
            for planner_name in planners:
                key = f"{planner_name}_{explainer_name}"
                if key in planner_features[env_name]:
                    planner_feature_sets.append((planner_name, planner_features[env_name][key]))
            
            # Calculate pairwise Jaccard similarities
            similarities = []
            for i in range(len(planner_feature_sets)):
                for j in range(i+1, len(planner_feature_sets)):
                    p1_name, p1_features = planner_feature_sets[i]
                    p2_name, p2_features = planner_feature_sets[j]
                    sim = jaccard_similarity(p1_features, p2_features)
                    similarities.append(sim)
            
            # Calculate average similarity (robustness)
            robustness = sum(similarities) / len(similarities) if similarities else 0
            
            # Update DataFrame with robustness values
            mask = (df["Environment"] == env_name) & (df["Explanation"] == explainer_name)
            df.loc[mask, "Planner_Robustness"] = robustness
    
    # Save results to CSV using pandas
    if args.average_only:
        # Group by all columns except Environment and compute averages
        avg_df = df.groupby([
            "Explanation", "Planner", "Num_Perturbations", 
            "Perturbation_Type"
        ]).agg({
            "Explanation_Stability": "mean",
            "Faithfulness_Score": "mean",
            "Planner_Robustness": "mean",
            "Path_Length": "mean",
            "Explanation_Time_s": "mean"
        }).reset_index()
        
        # Check if output file exists and append if it does
        if os.path.exists(args.output):
            # Read existing results
            existing_df = pd.read_csv(args.output)
            # Combine with new results
            combined_df = pd.concat([existing_df, avg_df], ignore_index=True)
            # Remove duplicates if any (keeping the latest entry)
            combined_df = combined_df.drop_duplicates(
                subset=["Explanation", "Planner", "Num_Perturbations", "Perturbation_Type"], 
                keep='last'
            )
            # Save combined results
            combined_df.to_csv(args.output, index=False)
            print(f"Results appended to existing file {args.output}")
            print(f"Total rows in combined file: {len(combined_df)}")
        else:
            # Save as new file
            avg_df.to_csv(args.output, index=False)
            print(f"New results file created at {args.output}")
        
        print(f"Rows in current results: {len(avg_df)}")
        print(f"DataFrame shape: {avg_df.shape}")
        
        # Display summary statistics
        print("\nSummary Statistics (Averaged):")
        print(avg_df.describe())
    else:
        # For full results (not averaged)
        if os.path.exists(args.output):
            # Read existing results
            existing_df = pd.read_csv(args.output)
            # Combine with new results
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates if any (based on all fields except the metrics)
            combined_df = combined_df.drop_duplicates(
                subset=["Environment", "Explanation", "Planner", "Num_Perturbations", "Perturbation_Type"],
                keep='last'
            )
            # Save combined results
            combined_df.to_csv(args.output, index=False)
            print(f"Results appended to existing file {args.output}")
            print(f"Total rows in combined file: {len(combined_df)}")
        else:
            # Save as new file
            df.to_csv(args.output, index=False)
            print(f"New results file created at {args.output}")
        
        print(f"Rows in current results: {len(df)}")
        print(f"DataFrame shape: {df.shape}")
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print(df.describe())

def main_loop():
    parser = argparse.ArgumentParser(description="Run comparison experiments for path planning explanations")
    parser.add_argument("--env_dir", type=str, default="environments/infeasible",
                        help="Directory with existing environments")
    parser.add_argument("--num_envs", type=int, default=100,
                        help="Number of environments to load")
    parser.add_argument("--output", type=str, default="why_did_you_fail_results.csv",
                        help="Output CSV file for results")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top features to consider")
    parser.add_argument("--average_only", default=True, action="store_true",
                        help="When set, only save average results across all environments")
    
    args = parser.parse_args()
    
    # Set up experiment parameters
    explanations = ["SHAP", "LIME"] #, "Anchors"]
    planners = ["A*", "Dijkstra", "Theta*", "BFS", "DFS", "Greedy Best-First"]
    perturbation_counts = [10, 25, 50, 75, 100, 125, 150, 175, 200]
    perturbation_types = ["remove", "move"] #, "random"] # random doesn't make sense, remove and move cover all cases
    
    MAX_PERTURBATION_RETRIES = 100  # Maximum attempts to find an infeasible perturbation
    NUM_ENVS = 10000

    for env_size in range(15, 16):
        for obstacle_num in range(5, 16, 2):

            # Initialize the experiment runner
            runner = BatchExperimentRunner()
            
            # Initialize results list for pandas DataFrame
            results_data = []

            OUTPUT = f"why_did_you_fail_results_grid_{env_size}_obstacles_{obstacle_num}.csv"
            
            # Load environments
            #env_files = [f for f in os.listdir(args.env_dir) if f.endswith('.json') and 'infeasible' in f]
            ENV_DIR = "environments/"+f"grid_{env_size}_obstacles_{obstacle_num}"+"/infeasible"
            env_files = [f for f in os.listdir(ENV_DIR) if f.endswith('.json') and 'infeasible' in f]
            if not env_files:
                print(f"No infeasible environments found in environments/grid_{env_size}_obstacles_{obstacle_num}/infeasible")
                return
            
            # Take the first num_envs environments or all if fewer available
            env_files = env_files[:NUM_ENVS]
            env_paths = [os.path.join(ENV_DIR, f) for f in env_files]
            
            print(f"Loaded {len(env_paths)} infeasible environments")
            
            # For storing planner robustness data
            planner_features = {}
            
            # Calculate total number of iterations for progress tracking
            total_iterations = len(env_paths) * len(planners) * len(explanations) * len(perturbation_types) * len(perturbation_counts)
            
            # Run experiments with a single progress bar
            with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
                for env_path in env_paths:
                    env_name = os.path.basename(env_path)
                    env = runner.load_environment(env_path)
                    
                    # Store features for each planner for this environment
                    planner_features[env_name] = {}
                    
                    # For each planner
                    for planner_name in planners:
                        planner_class = runner.planners[planner_name]
                        planner_instance = planner_class()
                        planner_instance.set_environment(
                            start=env.agent_pos,
                            goal=env.goal_pos,
                            grid_size=env.grid_size,
                            obstacles=env.obstacles
                        )
                        
                        # Get original path (should be None/empty for infeasible environments)
                        result = planner_instance.plan(return_steps=False)
                        original_path = result[0] if isinstance(result, tuple) else result
                        # for infeasible environments path length is 0
                        path_length = len(original_path) if original_path else 0
                        
                        # For each explanation method
                        for explainer_name in explanations:
                            explainer_class = runner.explainers[explainer_name]
                            explainer = explainer_class()
                            explainer.set_environment(env, planner_instance)
                            
                            # Get important features for original environment
                            original_features, explanation_time = get_important_features(
                                explainer, explainer_name, env, planner_instance, top_k=args.top_k
                            )
                            
                            # Store features for planner robustness calculation
                            planner_features[env_name][f"{planner_name}_{explainer_name}"] = original_features
                            
                            # Get ranked list of all obstacles for faithfulness calculation
                            ranked_obstacles = get_ranked_obstacles_from_explanation(explainer, explainer_name, env)
                            
                            # Calculate faithfulness score
                            faithfulness_score = calculate_faithfulness_score(env, planner_class, explainer, ranked_obstacles)
                            
                            # For each perturbation type and count
                            for perturbation_type in perturbation_types:
                                for num_perturbations in perturbation_counts:
                                    
                                    current_perturbed_env = None
                                    final_planner_check_for_perturbed = None
                                    found_infeasible_perturbation = False

                                    for attempt in range(MAX_PERTURBATION_RETRIES):
                                        # Create perturbed environment
                                        perturbed_env_candidate = perturb_environment(env, perturbation_type, num_perturbations)
                                        
                                        # Check if perturbed environment is still infeasible
                                        planner_check = planner_class()
                                        planner_check.set_environment(
                                            start=perturbed_env_candidate.agent_pos,
                                            goal=perturbed_env_candidate.goal_pos,
                                            grid_size=perturbed_env_candidate.grid_size,
                                            obstacles=perturbed_env_candidate.obstacles
                                        )
                                        result_check = planner_check.plan(return_steps=False)
                                        path_check = result_check[0] if isinstance(result_check, tuple) else result_check
                                        
                                        if not (path_check and len(path_check) > 0): # Path NOT found (still infeasible)
                                            current_perturbed_env = perturbed_env_candidate
                                            final_planner_check_for_perturbed = planner_check
                                            found_infeasible_perturbation = True
                                            break # Exit retry loop, we found a suitable environment
                                        # else: environment became feasible, try perturbing again in the next attempt
                                    
                                    if not found_infeasible_perturbation:
                                        print(f"Warning: Max retries ({MAX_PERTURBATION_RETRIES}) reached for {env_name}, {planner_name}, {explainer_name}, {perturbation_type}, {num_perturbations}. "
                                            f"Could not generate an infeasible perturbed environment. Skipping this combination.")
                                        pbar.update(1)  # Update progress bar even when skipped
                                        continue # Skip to the next num_perturbations or perturbation_type
                                    
                                    # Now, current_perturbed_env is an infeasible perturbed environment
                                    # and final_planner_check_for_perturbed is the planner instance that confirmed it.
                                    
                                    # Get explanation for perturbed environment
                                    explainer_perturbed = explainer_class()
                                    explainer_perturbed.set_environment(current_perturbed_env, final_planner_check_for_perturbed)
                                    perturbed_features, _ = get_important_features(
                                        explainer_perturbed, explainer_name, current_perturbed_env, final_planner_check_for_perturbed, top_k=args.top_k
                                    )
                                    
                                    # Calculate explanation stability (Jaccard similarity)
                                    stability = calculate_explanation_stability(original_features, perturbed_features)
                                    
                                    # Add results to list
                                    results_data.append({
                                        "Environment": env_name,
                                        "Explanation": explainer_name,
                                        "Planner": planner_name,
                                        "Num_Perturbations": num_perturbations,
                                        "Perturbation_Type": perturbation_type,
                                        "Explanation_Stability": stability,
                                        "Faithfulness_Score": faithfulness_score,
                                        "Planner_Robustness": None,  # Will be filled later
                                        "Path_Length": path_length,
                                        "Explanation_Time_s": explanation_time
                                    })
                                    
                                    # Update the progress bar
                                    pbar.update(1)
            
            # Create DataFrame from results
            df = pd.DataFrame(results_data)
            
            # Calculate planner robustness for each environment and explainer
            for env_name in planner_features:
                for explainer_name in explanations:
                    # Get features for each planner with this explainer
                    planner_feature_sets = []
                    for planner_name in planners:
                        key = f"{planner_name}_{explainer_name}"
                        if key in planner_features[env_name]:
                            planner_feature_sets.append((planner_name, planner_features[env_name][key]))
                    
                    # Calculate pairwise Jaccard similarities
                    similarities = []
                    for i in range(len(planner_feature_sets)):
                        for j in range(i+1, len(planner_feature_sets)):
                            p1_name, p1_features = planner_feature_sets[i]
                            p2_name, p2_features = planner_feature_sets[j]
                            sim = jaccard_similarity(p1_features, p2_features)
                            similarities.append(sim)
                    
                    # Calculate average similarity (robustness)
                    robustness = sum(similarities) / len(similarities) if similarities else 0
                    
                    # Update DataFrame with robustness values
                    mask = (df["Environment"] == env_name) & (df["Explanation"] == explainer_name)
                    df.loc[mask, "Planner_Robustness"] = robustness
            
            # Save results to CSV using pandas
            if args.average_only:
                # Group by all columns except Environment and compute averages
                avg_df = df.groupby([
                    "Explanation", "Planner", "Num_Perturbations", 
                    "Perturbation_Type"
                ]).agg({
                    "Explanation_Stability": "mean",
                    "Faithfulness_Score": "mean",
                    "Planner_Robustness": "mean",
                    "Path_Length": "mean",
                    "Explanation_Time_s": "mean"
                }).reset_index()
                
                # Check if output file exists and append if it does
                if os.path.exists(OUTPUT):
                    # Read existing results
                    existing_df = pd.read_csv(OUTPUT)
                    # Combine with new results
                    combined_df = pd.concat([existing_df, avg_df], ignore_index=True)
                    # Remove duplicates if any (keeping the latest entry)
                    combined_df = combined_df.drop_duplicates(
                        subset=["Explanation", "Planner", "Num_Perturbations", "Perturbation_Type"], 
                        keep='last'
                    )
                    # Save combined results
                    combined_df.to_csv(OUTPUT, index=False)
                    print(f"Results appended to existing file {OUTPUT}")
                    print(f"Total rows in combined file: {len(combined_df)}")
                else:
                    # Save as new file
                    avg_df.to_csv(OUTPUT, index=False)
                    print(f"New results file created at {OUTPUT}")
                
                print(f"Rows in current results: {len(avg_df)}")
                print(f"DataFrame shape: {avg_df.shape}")
                
                # Display summary statistics
                print("\nSummary Statistics (Averaged):")
                print(avg_df.describe())
            else:
                # For full results (not averaged)
                if os.path.exists(OUTPUT):
                    # Read existing results
                    existing_df = pd.read_csv(OUTPUT)
                    # Combine with new results
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    # Remove duplicates if any (based on all fields except the metrics)
                    combined_df = combined_df.drop_duplicates(
                        subset=["Environment", "Explanation", "Planner", "Num_Perturbations", "Perturbation_Type"],
                        keep='last'
                    )
                    # Save combined results
                    combined_df.to_csv(OUTPUT, index=False)
                    print(f"Results appended to existing file {OUTPUT}")
                    print(f"Total rows in combined file: {len(combined_df)}")
                else:
                    # Save as new file
                    df.to_csv(OUTPUT, index=False)
                    print(f"New results file created at {OUTPUT}")
                
                print(f"Rows in current results: {len(df)}")
                print(f"DataFrame shape: {df.shape}")
                
                # Display summary statistics
                print("\nSummary Statistics:")
                print(df.describe())

if __name__ == "__main__":
    #main()
    main_loop()
