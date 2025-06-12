import numpy as np
import random
from sklearn.linear_model import Ridge

class LimeExplainer:
    """
    LIME-based explanation for path planning.
    Explains the importance of obstacles by removing them and observing path changes.
    """
    
    def __init__(self):
        """Initialize the LIME explainer"""
        self.grid_size = None
        self.planner = None
        self.env = None
    
    def set_environment(self, env, planner):
        """
        Set the environment and planner to be explained
        
        Args:
            env: Environment object with grid_size and obstacles
            planner: Path planner that has a plan() method
        """
        self.env = env.clone()
        self.planner = planner
        self.grid_size = env.grid_size
    
    def explain(self, num_samples=100, callback=None, strategy="remove_each_obstacle_once", perturbation_mode="remove"):
        """
        Generate LIME-based explanations for the path planning problem
        
        Args:
            num_samples: Number of perturbation samples to generate
            callback: Optional callback function to update progress
            
        Returns:
            importance: Array of importance values for each obstacle shape
        """
        # Get obstacle keys
        obstacle_keys = list(self.env.obstacle_shapes.keys())
        num_obstacles = len(obstacle_keys)
        
        if num_obstacles == 0:
            return []
        
        combinations = []
        random_combinations = []
        # Generate perturbation combinations based on strategy
        if strategy == "remove_each_obstacle_once":
            combinations = self.env.generate_perturbation_combinations("remove_each_obstacle_once")
        
            # Also generate some random combinations (but limit total to avoid long processing)
            max_random = min(20, num_samples - len(combinations))
            for _ in range(max_random):
                combo = [random.randint(0, 1) for _ in range(num_obstacles)]
                random_combinations.append(combo)

        elif strategy == "random":
            combinations = self.env.generate_perturbation_combinations("random") * num_samples

        elif strategy == "full_combinations":
            combinations = self.env.generate_perturbation_combinations("full_combinations")
        
        # Combine all the combinations
        all_combinations = combinations + random_combinations
        
        X = []  # Perturbations (binary mask per sample)
        y = []  # Path costs per sample
        
        # Store original obstacles to restore at the end
        original_obstacles = self.env.obstacles.copy()
        
        # Process each combination
        total_combinations = len(all_combinations)
        
        for i, combination in enumerate(all_combinations):
            # Update progress if callback provided
            if callback:
                callback(i, total_combinations)
            
            # Ensure combination length matches current obstacle count
            current_obstacle_keys = list(self.env.obstacle_shapes.keys())
            current_num_obstacles = len(current_obstacle_keys)
            
            if len(combination) != current_num_obstacles:
                if len(combination) > current_num_obstacles:
                    # Trim combination if too long
                    combination = combination[:current_num_obstacles]
                else:
                    # Extend combination if too short
                    combination = combination + [1] * (current_num_obstacles - len(combination))
            
            # Apply perturbation using the adjusted combination
            original_state, _ = self.env.generate_perturbation(
                combination=combination,
                mode=perturbation_mode
            )
            
            # Run path planning
            path = self.planner.plan(
                self.env.agent_pos, 
                self.env.goal_pos, 
                self.env.obstacles
            )
            
            # Measure outcome: path length or failure penalty
            if path:
                path_length = len(path)
                success = True
            else:
                path_length = self.grid_size * 2  # Penalty for no path
                success = False
            
            # Record results
            X.append(combination)
            y.append(path_length)
            
            # Restore environment to original state
            self.env.restore_from_perturbation(original_state)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Fit a Ridge regression model to explain obstacle importance
        explainer = Ridge(alpha=1.0)
        explainer.fit(X, y)
        
        # Get coefficients - positive means removing obstacle increases cost (important)
        # Negative means removing obstacle decreases cost (harmful for path planning)
        importance = explainer.coef_
        
        # Replace any infinite values with large finite values to avoid rendering errors
        importance = np.nan_to_num(importance, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # Restore the original environment (just to be safe)
        self.env.obstacles = original_obstacles.copy()
        
        return importance