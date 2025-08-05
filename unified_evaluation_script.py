
import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from environment_generator import generate_environment
from shap_explainer import SHAPExplainer
from lime_explainer import LIMEExplainer
from cose_explainer import COSEExplainer
from astar import AStarPlanner
from grid_world_env import GridWorldEnv

# Define evaluation metrics
def compute_faithfulness(env, planner, explainer_output, top_k=3):
    """Remove top-k obstacles and check if planning succeeds"""
    sorted_obs = sorted(explainer_output.items(), key=lambda x: -x[1])[:top_k]
    env_copy = env.copy()
    for obs_id, _ in sorted_obs:
        env_copy.remove_obstacle_shape(obs_id)
    path = planner(env_copy)
    return path is not None

def compute_stability(explanations_list):
    """Compute average Jaccard similarity between multiple explanations"""
    def jaccard(set1, set2):
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 1.0
    if len(explanations_list) < 2:
        return 1.0
    scores = []
    for i in range(len(explanations_list) - 1):
        scores.append(jaccard(set(explanations_list[i]), set(explanations_list[i+1])))
    return sum(scores) / len(scores)

def compute_removal_efficiency(cose_set):
    return len(cose_set)

def run_experiment(env_id, grid_size, num_obstacles, planner, affordance):
    # Generate environment
    env = generate_environment(grid_size=grid_size, num_obstacles=num_obstacles, seed=env_id)
    planner_obj = AStarPlanner()
    path = planner_obj(env)
    if path is not None:
        return None  # Skip successful environments

    results = []

    for explainer_name in ["shap", "lime", "cose"]:
        start_time = time.time()

        if explainer_name == "shap":
            explainer = SHAPExplainer(env, planner_obj, affordance)
            explanation = explainer.explain()
            explanation_set = set(k for k, v in explanation.items() if v > 0)
        elif explainer_name == "lime":
            explainer = LIMEExplainer(env, planner_obj, affordance)
            explanation = explainer.explain()
            explanation_set = set(k for k, v in explanation.items() if v > 0)
        else:  # COSE
            explainer = COSEExplainer(env, planner_obj, affordance)
            explanation_set = explainer.explain()

        end_time = time.time()

        results.append({
            "Environment": env_id,
            "Planner": "AStar",
            "Explanation": explainer_name,
            "Question": "Q1",
            "Faithfulness_Score": compute_faithfulness(env.copy(), planner_obj, explanation),
            "Explanation_Stability": None,
            "Planner_Robustness": None,
            "Obstacle_Removal_Efficiency": None,
            "Path_Optimality": None,
            "Explanation_Time_s": round(end_time - start_time, 3)
        })

        results.append({
            "Environment": env_id,
            "Planner": "AStar",
            "Explanation": explainer_name,
            "Question": "Q2",
            "Faithfulness_Score": None,
            "Explanation_Stability": None,
            "Planner_Robustness": None,
            "Obstacle_Removal_Efficiency": compute_removal_efficiency(explanation_set) if explainer_name == "cose" else None,
            "Path_Optimality": None,
            "Explanation_Time_s": round(end_time - start_time, 3)
        })

        # Stability across random seeds
        if explainer_name in ["shap", "lime"]:
            variants = []
            for _ in range(3):
                explanation_variation = explainer.explain()
                variants.append(set(k for k, v in explanation_variation.items() if v > 0))
            stability_score = compute_stability(variants)
        else:
            stability_score = 1.0

        results.append({
            "Environment": env_id,
            "Planner": "AStar",
            "Explanation": explainer_name,
            "Question": "Q3",
            "Faithfulness_Score": None,
            "Explanation_Stability": stability_score,
            "Planner_Robustness": None,
            "Obstacle_Removal_Efficiency": None,
            "Path_Optimality": None,
            "Explanation_Time_s": round(end_time - start_time, 3)
        })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="evaluation_results.csv")
    parser.add_argument("--grid_sizes", nargs="+", type=int, default=[10, 15])
    parser.add_argument("--num_obstacles", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--affordance", type=str, choices=["remove", "move"], default="remove")
    parser.add_argument("--num_envs", type=int, default=10)
    args = parser.parse_args()

    all_results = []
    env_id = 0

    for size in args.grid_sizes:
        for num_obs in args.num_obstacles:
            for _ in tqdm(range(args.num_envs), desc=f"{size}x{size}_{num_obs}obs"):
                results = run_experiment(env_id, grid_size=size, num_obstacles=num_obs,
                                         planner=AStarPlanner(), affordance=args.affordance)
                if results:
                    all_results.extend(results)
                env_id += 1

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
