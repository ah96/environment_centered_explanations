import numpy as np
import random
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) explainer for path planning.
    
    Calculates feature importance using Shapley values from cooperative game theory.
    Each obstacle is a feature that contributes to the path planning outcome.
    """
    
    def __init__(self):
        """Initialize the SHAP explainer"""
        self.env = None
        self.planner = None
        self.grid_size = None
        self.baseline_path = None
        self.baseline_path_length = None
        
    def set_environment(self, env, planner):
        """
        Set the environment and planner to be explained
        
        Args:
            env: Environment object with grid_size and obstacles
            planner: Path planner that has a plan() method
        """
        self.env = env
        self.planner = planner
        self.grid_size = env.grid_size
        
        # Get baseline path (with all obstacles)
        self.baseline_path = planner.plan()
        self.baseline_path_length = len(self.baseline_path) if self.baseline_path else float('inf')
        
    def explain(self, num_samples=100, callback=None):
        """
        Generate SHAP-based explanations for the path planning problem
        
        Args:
            num_samples: Number of samples to use for approximating Shapley values
            callback: Optional callback function to update progress
            
        Returns:
            dict: Dictionary with SHAP values for each obstacle shape
        """
        # Get obstacle shapes
        obstacle_keys = list(self.env.obstacle_shapes.keys())
        num_obstacles = len(obstacle_keys)
        
        if num_obstacles == 0:
            return {}
            
        # Initialize SHAP values
        shap_values = {shape_id: 0 for shape_id in obstacle_keys}
        
        # Store original obstacles for restoration
        original_obstacles = self.env.obstacles.copy()
        
        # First, measure the path with no obstacles
        no_obstacles_path_length = self.compute_path_length([0] * num_obstacles)
        
        # Then measure the path with all obstacles 
        all_obstacles_path_length = self.compute_path_length([1] * num_obstacles)
        
        # Track which combinations we've already evaluated
        evaluated_combinations = {}
        
        # We'll use a sampling-based approach for approximation
        for sample in range(num_samples):
            if callback:
                callback(sample, num_samples)
                
            # Generate a random permutation of obstacles
            obstacle_order = list(obstacle_keys)
            random.shuffle(obstacle_order)
            
            # Start with no obstacles
            current_combination = [0] * num_obstacles
            prev_path_length = no_obstacles_path_length
            
            # Add obstacles one by one in the random order
            for obs_idx, obstacle_id in enumerate(obstacle_order):
                # Get the index in our feature vector
                idx = obstacle_keys.index(obstacle_id)
                
                # Add this obstacle
                new_combination = current_combination.copy()
                new_combination[idx] = 1
                
                # Compute new path length
                new_path_length = self.compute_path_length(new_combination, evaluated_combinations)
                
                # The marginal contribution is the change in path length
                # For path planning, a positive contribution means the obstacle makes the path longer (worse)
                marginal_contribution = new_path_length - prev_path_length
                
                # Update SHAP value for this obstacle
                shap_values[obstacle_id] += marginal_contribution / num_samples
                
                # Update for next iteration
                current_combination = new_combination
                prev_path_length = new_path_length
        
        # Debug: print min/max values to verify we're calculating real values
        if shap_values:
            all_vals = list(shap_values.values())
            print(f"SHAP value range: min={min(all_vals)}, max={max(all_vals)}")
            
        # Restore the original environment
        self.env.obstacles = original_obstacles.copy()
        
        return shap_values
        
    def compute_path_length(self, combination, cache=None):
        """
        Compute the path length for a given obstacle combination
        
        Args:
            combination: Binary vector indicating which obstacles to include (1=keep, 0=remove)
            cache: Optional cache of already computed results
            
        Returns:
            float: Path length, or infinity if no path exists
        """
        # Check cache first
        if cache is not None:
            key = tuple(combination)
            if key in cache:
                return cache[key]
        
        # Apply the combination to the environment
        original_state, _ = self.env.generate_perturbation(combination=combination)
        
        # Run the planner
        path = self.planner.plan()
        
        # Compute path length
        path_length = len(path) if path else float('inf')
        
        # Restore environment
        self.env.restore_from_perturbation(original_state)
        
        # Cache the result
        if cache is not None:
            cache[tuple(combination)] = path_length
            
        return path_length
        
    def visualize(self, shap_values):
        """
        Visualize SHAP values on the grid
        
        Args:
            shap_values: Dictionary of SHAP values from explain()
            
        Returns:
            matplotlib figure
        """
        if not shap_values:
            return None
            
        # Create a grid to visualize SHAP values
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Find min and max SHAP values for normalization
        all_values = list(shap_values.values())
        if all_values:
            min_shap = min(all_values)
            max_shap = max(all_values)
            max_abs_shap = max(abs(min_shap), abs(max_shap))
            
            # Avoid division by zero
            if max_abs_shap == 0:
                max_abs_shap = 1
        else:
            max_abs_shap = 1
        
        # Map SHAP values to the grid
        for shape_id, value in shap_values.items():
            for pos in self.env.obstacle_shapes[shape_id]:
                x, y = pos
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[x, y] = value
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use diverging colormap
        cmap = plt.cm.RdBu_r
        
        # Create the heatmap
        vmax = max_abs_shap if max_abs_shap > 0 else 1
        vmin = -vmax
        heatmap = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # Add obstacle labels with their SHAP values
        for shape_id, value in shap_values.items():
            points = self.env.obstacle_shapes[shape_id]
            if points:
                first_point = points[0]
                ax.annotate(f"#{shape_id}: {value:.2f}", 
                        (first_point[1], first_point[0]),
                        color='black', fontsize=8,
                        ha='center', va='center')
        
        # Add title
        ax.set_title('SHAP Values: Impact of Obstacles on Path Length\n' +
                    'Blue: Obstacle increases path length (obstructive)\n' +
                    'Red: Obstacle decreases path length (helpful)')
        
        # Mark start and goal positions
        if self.env.agent_pos:
            ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0],
                    color='blue', s=150, marker='o', label='Start')
        if self.env.goal_pos:
            ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0],
                    color='green', s=150, marker='*', label='Goal')
            
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        # REMOVE THE INVERSION TO FIX THE UPSIDE-DOWN ISSUE
        # ax.invert_yaxis()  # <-- Comment out or remove this line
        
        return fig