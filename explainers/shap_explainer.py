import numpy as np
import random
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

class SHAPExplainer:
    def __init__(self):
        self.env = None
        self.planner = None
        self.grid_size = None
        self.baseline_path = None
        self.baseline_path_length = None

    def set_environment(self, env, planner):
        self.env = env.clone()
        self.planner = planner
        self.grid_size = env.grid_size
        self.baseline_path = planner.plan()
        self.baseline_path_length = len(self.baseline_path) if self.baseline_path else self.get_penalty_value()

    def explain(self, num_samples=100, callback=None, perturbation_mode="remove"):
        # Save original obstacle state at the beginning
        original_obstacles = copy.deepcopy(self.env.obstacles)
        original_obstacle_shapes = copy.deepcopy(self.env.obstacle_shapes)
        
        # Get fixed obstacle keys that won't change during computation
        obstacle_keys = list(original_obstacle_shapes.keys())
        num_obstacles = len(obstacle_keys)

        if num_obstacles == 0:
            return {}

        # Initialize SHAP values with fixed obstacle keys
        shap_values = {shape_id: 0 for shape_id in obstacle_keys}
        evaluated_combinations = {}

        baseline_path_length = self.compute_path_length([1] * num_obstacles, evaluated_combinations, perturbation_mode)

        for sample in range(num_samples):
            if callback:
                callback(sample, num_samples)

            #-----DEBUGGING-----
            if len(shap_values) > 15: # len(self.env.obstacle_shapes):
                raise ValueError(f"shape_id {shape_id} is out of range for obstacle_shapes (len={len(self.env.obstacle_shapes)})")
            #-------------------
                
            # Reset environment to original state for each sample
            self.env.obstacles = copy.deepcopy(original_obstacles)
            self.env.obstacle_shapes = copy.deepcopy(original_obstacle_shapes)

            obstacle_order = list(range(num_obstacles))
            random.shuffle(obstacle_order)

            current_combination = [1] * num_obstacles
            prev_path_length = baseline_path_length

            for obs_idx in obstacle_order:
                current_combination[obs_idx] = 0
                new_path_length = self.compute_path_length(current_combination, evaluated_combinations, perturbation_mode)
                marginal_contribution = prev_path_length - new_path_length
                shap_values[obstacle_keys[obs_idx]] += marginal_contribution
                prev_path_length = new_path_length

        # Calculate average SHAP values using original obstacle keys
        for shape_id in obstacle_keys:
            shap_values[shape_id] /= num_samples

        # Restore environment to original state
        self.env.obstacles = copy.deepcopy(original_obstacles)
        self.env.obstacle_shapes = copy.deepcopy(original_obstacle_shapes)

        if shap_values:
            all_vals = list(shap_values.values())
            print(f"SHAP value range: min={min(all_vals)}, max={max(all_vals)}")

        print("\n[SHAP DEBUG] Baseline path length (all obstacles):", baseline_path_length)
        for shape_id in obstacle_keys:
            combo = [1] * num_obstacles
            idx = obstacle_keys.index(shape_id)
            combo[idx] = 0
            length = self.compute_path_length(combo, None, perturbation_mode)
            print(f"Removing obstacle #{shape_id}: path length = {length}")
                

        for k, v in shap_values.items():
            print(f"Obstacle #{k}: SHAP value = {v:.2f}")

        return shap_values

    def compute_path_length(self, combination, cache=None, perturbation_mode="remove"):
        if cache is not None:
            key = tuple(combination)
            if key in cache:
                return cache[key]

        # Create a completely new environment clone for this computation
        env_clone = self.env.clone()
        
        # Get current obstacle IDs from the clone
        all_shape_ids = list(env_clone.obstacle_shapes.keys())
        
        # Ensure combination length matches current obstacle count
        if len(combination) != len(all_shape_ids):
            if len(combination) > len(all_shape_ids):
                combination = combination[:len(all_shape_ids)]
            else:
                combination = combination + [1] * (len(all_shape_ids) - len(combination))

        # Apply the perturbation to the clone
        for i, val in enumerate(combination):
            if i < len(all_shape_ids) and val == 0:  # Remove this obstacle
                shape_id = all_shape_ids[i]
                if shape_id in env_clone.obstacle_shapes:
                    points_to_remove = env_clone.obstacle_shapes[shape_id]
                    env_clone.obstacles = [p for p in env_clone.obstacles if p not in points_to_remove]
                    del env_clone.obstacle_shapes[shape_id]

        # Use the cloned environment for path calculation
        planner_class = type(self.planner)
        planner = planner_class()
        planner.set_environment(
            start=env_clone.agent_pos,
            goal=env_clone.goal_pos,
            grid_size=env_clone.grid_size,
            obstacles=env_clone.obstacles
        )
        path = planner.plan()

        path_length = len(path) if path else self.get_penalty_value()
        
        # No need to restore - we used a clone
        
        if cache is not None:
            cache[tuple(combination)] = path_length

        return path_length
    
    def get_penalty_value(self):
        # Return a large constant penalty for infeasible paths
        return self.grid_size * self.grid_size + 1

    def visualize(self, shap_values):
        if not shap_values:
            return None

        grid = np.zeros((self.grid_size, self.grid_size))
        all_values = list(shap_values.values())
        max_abs_shap = max(abs(min(all_values)), abs(max(all_values))) if all_values else 1
        if max_abs_shap == 0:
            max_abs_shap = 1

        for shape_id, value in shap_values.items():
            if shape_id in self.env.obstacle_shapes:  # Check if shape_id exists
                for pos in self.env.obstacle_shapes[shape_id]:
                    x, y = pos
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        grid[x, y] = value

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.RdBu
        heatmap = ax.imshow(grid, cmap=cmap, vmin=-max_abs_shap, vmax=max_abs_shap)

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Annotate obstacle labels
        for shape_id, value in shap_values.items():
            if shape_id in self.env.obstacle_shapes and self.env.obstacle_shapes[shape_id]:
                points = self.env.obstacle_shapes[shape_id]
                if points:
                    fx, fy = points[0]
                    ax.annotate(f"#{shape_id}\n{value:.2f}",
                                (fy, fx),
                                color='black', fontsize=8,
                                ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_title('SHAP Values: Impact of Obstacles on Path Length\n'
                     'Blue = increases cost (obstructive), Red = decreases cost (helpful)')

        # Plot start and goal
        if self.env.agent_pos:
            ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0],
                       color='blue', s=150, marker='o', label='Start')
        if self.env.goal_pos:
            ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0],
                       color='green', s=150, marker='*', label='Goal')

        # Add colorbar
        cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8)
        cbar.set_label("SHAP Value (Impact on Path Length)", fontsize=10)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        return fig