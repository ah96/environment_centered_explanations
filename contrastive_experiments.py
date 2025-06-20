import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from explanations.contrastive_explainer import ContrastiveExplainer
from grid_world_env import GridWorldEnv
from path_planning.astar import AStarPlanner
from path_planning.dijkstra import DijkstraPlanner
from path_planning.bfs import BFSPlanner
from path_planning.theta_star import ThetaStarPlanner

# Supported planners
PLANNER_MAP = {
    "A*": AStarPlanner,
    "Dijkstra": DijkstraPlanner,
    "BFS": BFSPlanner,
    "Theta*": ThetaStarPlanner
}

# Supported perturbation strategies
PERTURBATION_MODES = ["remove", "move"]

def jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.
    Returns 1.0 if both sets are empty, else |A ∩ B| / |A ∪ B|.
    """
    return 1.0 if not set1 and not set2 else len(set1 & set2) / len(set1 | set2)

def load_environment_from_dict(env_data):
    """
    Load a GridWorldEnv instance from a dictionary (e.g., loaded from JSON).
    """
    env = GridWorldEnv(grid_size=env_data["grid_size"], num_obstacles=env_data["num_obstacles"])
    env.agent_pos = env_data["agent_pos"]
    env.goal_pos = env_data["goal_pos"]
    env.obstacle_shapes = {int(k): v for k, v in env_data["obstacle_shapes"].items()}
    env.update_obstacles_from_shapes()
    return env

def calculate_faithfulness(env, planner_cls, factual_path, contrastive_path, top_obstacles, perturbation_mode="remove"):
    """
    Compute faithfulness score as the minimum fraction of top-k obstacles
    that need to be perturbed (removed or moved) for the planner to switch
    from the factual to the contrastive path.
    """
    total = len(top_obstacles)
    if total == 0:
        return 0.0

    modified_env = env.clone()
    for i, obs_id in enumerate(top_obstacles, 1):
        if perturbation_mode == "remove":
            modified_env.remove_obstacle_shape(obs_id)
        elif perturbation_mode == "move":
            modified_env.move_obstacle_shape(obs_id)

        planner = planner_cls()
        planner.set_environment(
            start=modified_env.agent_pos,
            goal=modified_env.goal_pos,
            grid_size=modified_env.grid_size,
            obstacles=modified_env.obstacles
        )
        new_path = planner.plan()
        if new_path and new_path != factual_path and new_path == contrastive_path:
            return i / total
    return 1.0

def path_obstacle_overlap(path, env, obstacle_ids, threshold=0):
    """
    Compute how many of the explanatory obstacles lie near the path.
    Returns ratio of overlapping obstacles to total explanatory obstacles.
    """
    path_set = set(tuple(p) for p in path)
    overlap = 0
    for obs_id in obstacle_ids:
        shape = env.obstacle_shapes.get(obs_id, [])
        if any(abs(px - ox) + abs(py - oy) <= threshold for (px, py) in path_set for (ox, oy) in shape):
            overlap += 1
    return overlap / max(len(obstacle_ids), 1)

def main():
    """
    Main loop for evaluating contrastive explanations across environment pairs,
    planners, and perturbation strategies. Computes metrics including:
    - Faithfulness
    - Stability
    - Sparsity
    - Asymmetry
    - Relative Path Length Difference
    - Path-Obstacle Overlap
    Saves results as a CSV file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="contrastive_envs", help="Directory with contrastive pairs")
    parser.add_argument("--output_csv", type=str, default="contrastive_explanation_results.csv", help="CSV to store results")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k features to consider")
    args = parser.parse_args()

    results = []
    pairs = sorted([d for d in os.listdir(args.input_dir) if d.startswith("pair_")])

    for pair_dir in tqdm(pairs, desc="Evaluating contrastive explanations"):
        pair_path = os.path.join(args.input_dir, pair_dir, "pair.json")
        if not os.path.exists(pair_path):
            continue

        with open(pair_path, "r") as f:
            data = json.load(f)

        env_a = load_environment_from_dict(data["original_environment"])
        env_b = load_environment_from_dict(data["contrastive_environment"])
        path_a = data["original_environment"]["path"]
        path_b = data["contrastive_environment"]["path"]

        if not path_a or not path_b or path_a == path_b:
            continue

        for planner_name, planner_cls in PLANNER_MAP.items():
            for perturbation_mode in PERTURBATION_MODES:
                try:
                    planner = planner_cls()
                    explainer = ContrastiveExplainer(env=env_a, alt_env=env_b)
                    explainer.set_environment(env_a, planner)
                    explanation = explainer.explain(
                        factual_path=path_a,
                        contrastive_path=path_b,
                        minimal=True,
                        proximity_threshold=0,
                        perturbation_mode=perturbation_mode
                    )
                except Exception as e:
                    print(f"[!] Explainer failed for {planner_name} / {perturbation_mode} in {pair_dir}: {e}")
                    continue

                affected_ids = [o["id"] for o in explanation["obstacles_affecting_choice"]]
                top_k_obstacles = set(affected_ids[:args.top_k])

                # Faithfulness
                faithfulness = calculate_faithfulness(
                    env_a, planner_cls, path_a, path_b, list(top_k_obstacles),
                    perturbation_mode=perturbation_mode
                )

                # Stability (reverse explanation)
                try:
                    planner_b = planner_cls()
                    explainer_perturbed = ContrastiveExplainer(env=env_b, alt_env=env_a)
                    explainer_perturbed.set_environment(env_b, planner_b)
                    reverse_explanation = explainer_perturbed.explain(
                        factual_path=path_b,
                        contrastive_path=path_a,
                        minimal=True,
                        proximity_threshold=0,
                        perturbation_mode=perturbation_mode
                    )
                    perturbed_ids = {o["id"] for o in reverse_explanation["obstacles_affecting_choice"]}
                except Exception as e:
                    print(f"[!] Stability check failed for {planner_name} / {perturbation_mode} in {pair_dir}: {e}")
                    perturbed_ids = set()

                # Additional Metrics
                stability = jaccard_similarity(top_k_obstacles, perturbed_ids)
                sparsity = len(affected_ids)
                asymmetry = 1.0 - jaccard_similarity(set(affected_ids), perturbed_ids)
                relative_path_diff = (len(path_b) - len(path_a)) / max(len(path_a), 1)
                path_overlap = path_obstacle_overlap(path_a, env_a, affected_ids)

                results.append({
                    "Pair": pair_dir,
                    "Planner": planner_name,
                    "Perturbation_Mode": perturbation_mode,
                    "Top_k": args.top_k,
                    "Num_Affected": sparsity,
                    "Faithfulness_Score": faithfulness,
                    "Stability": stability,
                    "Path_A_Length": len(path_a),
                    "Path_B_Length": len(path_b),
                    "Explanation_Sparsity": sparsity,
                    "Explanation_Asymmetry": asymmetry,
                    "Relative_Path_Diff": relative_path_diff,
                    "Path_Overlap": path_overlap
                })

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"[✓] Results saved to {args.output_csv}")
    print(df.describe())

if __name__ == "__main__":
    main()
