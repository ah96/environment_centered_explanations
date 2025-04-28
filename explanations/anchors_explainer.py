import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class AnchorsExplainer:
    """
    Anchors-based explanation for path planning.
    
    Identifies obstacle configurations that consistently affect path planning outcomes.
    Anchors are rules that "lock" certain path planning behaviors.
    """
    
    def __init__(self):
        """Initialize the Anchors explainer"""
        self.env = None
        self.planner = None
        self.grid_size = None
        
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
        
    def explain(self, num_samples=100, precision_threshold=0.95, min_coverage=0.1, callback=None):
        """
        Generate Anchors-based explanations for the path planning problem
        
        Args:
            num_samples: Number of perturbation samples to generate
            precision_threshold: Minimum precision required for an anchor rule
            min_coverage: Minimum coverage required for an anchor rule  
            callback: Optional callback function to update progress
            
        Returns:
            dict: Dictionary of anchor rules and their metrics
        """
        # Get obstacle shapes
        obstacle_keys = list(self.env.obstacle_shapes.keys())
        num_obstacles = len(obstacle_keys)
        
        if num_obstacles == 0:
            return {}
            
        # Get baseline path (with all obstacles)
        original_path = self.planner.plan()
        baseline_path_length = len(original_path) if original_path else float('inf')
        
        # Generate candidate anchor rules (combinations of obstacles to keep or remove)
        candidate_anchors = []
        
        # Start with single obstacle rules (keep/remove one obstacle)
        for i in range(num_obstacles):
            # Rule to keep obstacle i
            keep_rule = [None] * num_obstacles
            keep_rule[i] = 1
            candidate_anchors.append(keep_rule)
            
            # Rule to remove obstacle i
            remove_rule = [None] * num_obstacles
            remove_rule[i] = 0
            candidate_anchors.append(remove_rule)
        
        # Add some random rules with 2-3 obstacles specified
        for _ in range(min(20, num_obstacles * 2)):
            rule_size = random.randint(2, min(3, num_obstacles))
            rule = [None] * num_obstacles
            for _ in range(rule_size):
                idx = random.randint(0, num_obstacles - 1)
                rule[idx] = random.choice([0, 1])
            candidate_anchors.append(rule)
            
        # Test each candidate anchor
        anchor_results = {}
        original_obstacles = self.env.obstacles.copy()
        
        # For progress tracking
        total_iterations = len(candidate_anchors) * num_samples
        iteration = 0
        
        for anchor_idx, anchor_rule in enumerate(candidate_anchors):
            if callback:
                callback(anchor_idx, len(candidate_anchors))
                
            # Generate samples that satisfy this anchor rule
            samples_satisfying_rule = []
            samples_results = []
            
            # Generate samples
            for _ in range(num_samples):
                # Create a random combination that satisfies the anchor rule
                combination = []
                for i in range(num_obstacles):
                    if anchor_rule[i] is not None:
                        combination.append(anchor_rule[i])  # Use the rule's value
                    else:
                        combination.append(random.randint(0, 1))  # Random value for unspecified
                        
                samples_satisfying_rule.append(combination)
                
                # Apply the combination
                original_state, _ = self.env.generate_perturbation(combination=combination)
                
                # Test the path
                path = self.planner.plan()
                
                # Determine if path changed significantly from baseline
                path_length = len(path) if path else float('inf')
                path_changed = abs(path_length - baseline_path_length) > 1  # Consider 1 step tolerance
                
                # Record result
                samples_results.append(path_changed)
                
                # Restore environment
                self.env.restore_from_perturbation(original_state)
                
                # Update progress
                iteration += 1
                if callback and iteration % 10 == 0:
                    callback(iteration, total_iterations)
            
            # Calculate precision (how often does the rule predict the outcome)
            if samples_results:
                # Count the most common outcome
                outcome_counts = defaultdict(int)
                for result in samples_results:
                    outcome_counts[result] += 1
                
                majority_outcome = max(outcome_counts.items(), key=lambda x: x[1])[0]
                precision = outcome_counts[majority_outcome] / len(samples_results)
                coverage = sum(1 for r in anchor_rule if r is not None) / num_obstacles
                
                # Only keep rules that meet our thresholds
                if precision >= precision_threshold and coverage >= min_coverage:
                    # Create a readable rule
                    rule_description = []
                    for i, value in enumerate(anchor_rule):
                        if value is not None:
                            action = "keep" if value == 1 else "remove"
                            rule_description.append(f"{action} obstacle #{i}")
                    
                    rule_str = " AND ".join(rule_description)
                    outcome_str = "path changes significantly" if majority_outcome else "path remains similar"
                    
                    anchor_results[rule_str] = {
                        "rule": anchor_rule,
                        "precision": precision, 
                        "coverage": coverage,
                        "outcome": majority_outcome,
                        "description": f"When we {rule_str}, the {outcome_str} (precision: {precision:.2f})"
                    }
        
        # Restore the original environment
        self.env.obstacles = original_obstacles.copy()
        
        # Sort results by precision and coverage
        sorted_anchors = dict(sorted(
            anchor_results.items(), 
            key=lambda x: (x[1]["precision"], x[1]["coverage"]), 
            reverse=True
        ))
        
        return sorted_anchors
        
    def visualize(self, anchors, max_anchors=5):
        """
        Visualize the found anchors
        
        Args:
            anchors: Dictionary of anchor rules from explain()
            max_anchors: Maximum number of anchors to visualize
        """
        if not anchors:
            return None
            
        # Get top anchors limited by max_anchors
        top_anchors = list(anchors.items())[:max_anchors]
        
        # Create figure for visualization
        fig, axes = plt.subplots(len(top_anchors), 1, figsize=(10, 4*len(top_anchors)))
        if len(top_anchors) == 1:
            axes = [axes]
            
        for i, (rule_str, anchor_data) in enumerate(top_anchors):
            ax = axes[i]
            
            # Highlight obstacles according to the anchor rule
            grid = np.zeros((self.grid_size, self.grid_size))
            
            obstacle_keys = list(self.env.obstacle_shapes.keys())
            rule = anchor_data["rule"]
            
            for idx, shape_id in enumerate(obstacle_keys):
                if idx < len(rule) and rule[idx] is not None:
                    # Color depends on rule (1=keep, 0=remove)
                    value = 1 if rule[idx] == 1 else -1
                    
                    for pos in self.env.obstacle_shapes[shape_id]:
                        x, y = pos
                        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                            grid[x, y] = value
            
            # Display the grid
            cmap = plt.cm.coolwarm
            heatmap = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1)
            
            # Add grid lines
            ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            
            # Remove tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add title with the rule description
            ax.set_title(f"Anchor #{i+1}: {anchor_data['description']}")
            
            # Mark start and goal positions
            if self.env.agent_pos:
                ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0], 
                        color='blue', s=150, marker='o', label='Start')
            if self.env.goal_pos:
                ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0], 
                        color='green', s=150, marker='*', label='Goal')
                
            # Add legend
            if i == 0:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
            
            # REMOVE THE INVERSION TO FIX THE UPSIDE-DOWN ISSUE
            # ax.invert_yaxis()  # <-- Comment out or remove this line
            
        plt.tight_layout()
        return fig