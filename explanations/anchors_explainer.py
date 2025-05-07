import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colormaps
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
        
    def explain(self, num_samples=100, precision_threshold=0.95, min_coverage=0.1, callback=None, detect_changes=False, perturbation_mode="remove"):
        """
        Generate Anchors-based explanations for the path planning problem.

        Args:
            num_samples: Number of perturbation samples to generate
            precision_threshold: Minimum precision required for an anchor rule
            min_coverage: Minimum coverage required for an anchor rule  
            callback: Optional callback function to update progress
            detect_changes: If True, find anchors that cause path changes.
                            If False (default), find anchors that preserve path behavior.

        Returns:
            dict: Dictionary of anchor rules and their metrics
        """
        obstacle_keys = list(self.env.obstacle_shapes.keys())
        num_obstacles = len(obstacle_keys)

        if num_obstacles == 0:
            return {}

        original_path = self.planner.plan()
        baseline_path_length = len(original_path) if original_path else float('inf')
        
        # Adjust parameters for path changes detection
        if detect_changes:
            # Lower threshold for path changes to find any relevant patterns
            precision_threshold = max(0.6, precision_threshold * 0.7)
            # Allow smaller coverage for path changes
            min_coverage = min_coverage * 0.8
        
        candidate_anchors = []

        # Basic one-obstacle rules
        for i in range(num_obstacles):
            keep_rule = [None] * num_obstacles
            keep_rule[i] = 1
            candidate_anchors.append(keep_rule)

            remove_rule = [None] * num_obstacles
            remove_rule[i] = 0
            candidate_anchors.append(remove_rule)

        # Random multi-obstacle rules - adjust max rule size for path changes
        max_rule_size = 5 if detect_changes else 3
        for _ in range(min(40, num_obstacles * 2)):
            rule_size = random.randint(2, min(max_rule_size, num_obstacles))
            rule = [None] * num_obstacles
            for _ in range(rule_size):
                idx = random.randint(0, num_obstacles - 1)
                rule[idx] = random.choice([0, 1])
            candidate_anchors.append(rule)
            
        # Add more specific combinations - especially useful for path changes
        for i in range(num_obstacles):
            for j in range(i+1, num_obstacles):
                # Try keep-keep combinations
                rule = [None] * num_obstacles
                rule[i] = 1
                rule[j] = 1
                candidate_anchors.append(rule)
                
                # Try remove-remove combinations
                rule = [None] * num_obstacles
                rule[i] = 0
                rule[j] = 0
                candidate_anchors.append(rule)
                
                # Try keep-remove combinations
                rule = [None] * num_obstacles
                rule[i] = 1
                rule[j] = 0
                candidate_anchors.append(rule)
                
                rule = [None] * num_obstacles
                rule[i] = 0
                rule[j] = 1
                candidate_anchors.append(rule)
        
        # For path changes, add larger combinations - clusters of obstacles
        if detect_changes and num_obstacles > 3:
            # Try some larger combinations for more complex scenarios
            for _ in range(min(20, num_obstacles)):
                # Create larger rules (3-6 obstacles specified)
                rule_size = random.randint(3, min(6, num_obstacles))
                rule = [None] * num_obstacles
                selected_indices = random.sample(range(num_obstacles), rule_size)
                for idx in selected_indices:
                    rule[idx] = random.choice([0, 1])
                candidate_anchors.append(rule)
                
            # Try removing/keeping clusters of adjacent obstacles
            # This is useful because path changes often happen when groups of obstacles are affected
            obstacle_indices = list(range(num_obstacles))
            random.shuffle(obstacle_indices)
            for start_idx in range(0, min(20, num_obstacles), 3):
                end_idx = min(start_idx + 3, num_obstacles)
                selected_indices = obstacle_indices[start_idx:end_idx]
                
                # Keep all selected
                rule = [None] * num_obstacles
                for idx in selected_indices:
                    rule[idx] = 1  # Keep
                candidate_anchors.append(rule)
                
                # Remove all selected
                rule = [None] * num_obstacles
                for idx in selected_indices:
                    rule[idx] = 0  # Remove
                candidate_anchors.append(rule)

        anchor_results = {}
        original_obstacles = self.env.obstacles.copy()
        total_iterations = len(candidate_anchors) * num_samples
        iteration = 0
        
        # If detecting path changes, run some initial tests to find promising obstacle combinations
        promising_indices = []
        if detect_changes and num_obstacles > 5:
            # Check each obstacle's individual impact
            for i in range(min(num_obstacles, 20)):
                # Remove just this obstacle
                temp_obstacles = original_obstacles.copy()
                obstacle_id = obstacle_keys[i]
                if obstacle_id in temp_obstacles:
                    del temp_obstacles[obstacle_id]
                    
                    self.env.obstacles = temp_obstacles
                    path = self.planner.plan()
                    new_path_length = len(path) if path else float('inf')
                    
                    # If removing this obstacle causes a significant path change, it's promising
                    if abs(new_path_length - baseline_path_length) > 1:
                        promising_indices.append(i)
                        
                    self.env.obstacles = original_obstacles.copy()

        for anchor_idx, anchor_rule in enumerate(candidate_anchors):
            if callback:
                callback(anchor_idx, len(candidate_anchors))
                
            # For path changes, prioritize rules that involve promising obstacles
            if detect_changes and promising_indices and random.random() < 0.3:
                # 30% chance to modify this rule to include promising obstacles
                rule = list(anchor_rule)  # Copy the rule
                promising_idx = random.choice(promising_indices)
                rule[promising_idx] = random.choice([0, 1])  # Either keep or remove this promising obstacle
                anchor_rule = rule

            samples_results = []

            for _ in range(num_samples):
                # Build a perturbation that satisfies this anchor rule
                combination = []
                for i in range(num_obstacles):
                    if anchor_rule[i] is not None:
                        combination.append(anchor_rule[i])
                    else:
                        combination.append(random.randint(0, 1))

                original_state, _ = self.env.generate_perturbation(combination=combination, mode=perturbation_mode)
                path = self.planner.plan()
                path_length = len(path) if path else float('inf')

                path_changed = abs(path_length - baseline_path_length) > 1
                samples_results.append(path_changed)

                self.env.restore_from_perturbation(original_state)
                iteration += 1
                if callback and iteration % 10 == 0:
                    callback(iteration, total_iterations)

            if samples_results:
                outcome_counts = defaultdict(int)
                for result in samples_results:
                    outcome_counts[result] += 1

                majority_outcome = max(outcome_counts.items(), key=lambda x: x[1])[0]
                precision = outcome_counts[majority_outcome] / len(samples_results)
                coverage = sum(1 for r in anchor_rule if r is not None) / num_obstacles

                # Modified anchor selection logic based on the detect_changes parameter
                if detect_changes:
                    # For path changes, we want rules where majority_outcome is True (path changed)
                    if majority_outcome and precision >= precision_threshold and coverage >= min_coverage:
                        rule_description = []
                        for i, value in enumerate(anchor_rule):
                            if value is not None:
                                action = "keep" if value == 1 else "remove"
                                rule_description.append(f"{action} obstacle #{obstacle_keys[i]}")
                        
                        rule_str = " AND ".join(rule_description)
                        anchor_results[rule_str] = {
                            "rule": anchor_rule,
                            "precision": precision, 
                            "coverage": coverage,
                            "outcome": majority_outcome,
                            "description": f"When we {rule_str}, the path changes significantly (precision: {precision:.2f})"
                        }
                else:
                    # For path stability, we want rules where majority_outcome is False (path didn't change)
                    if not majority_outcome and precision >= precision_threshold and coverage >= min_coverage:
                        rule_description = []
                        for i, value in enumerate(anchor_rule):
                            if value is not None:
                                action = "keep" if value == 1 else "remove"
                                rule_description.append(f"{action} obstacle #{obstacle_keys[i]}")
                        
                        rule_str = " AND ".join(rule_description)
                        anchor_results[rule_str] = {
                            "rule": anchor_rule,
                            "precision": precision, 
                            "coverage": coverage,
                            "outcome": majority_outcome,
                            "description": f"When we {rule_str}, the path remains similar (precision: {precision:.2f})"
                        }

        self.env.obstacles = original_obstacles.copy()

        sorted_anchors = dict(sorted(
            anchor_results.items(),
            key=lambda x: (x[1]["precision"], x[1]["coverage"]),
            reverse=True
        ))

        return sorted_anchors

        
    def visualize(self, anchors, max_anchors=5):
        """
        Visualize the found anchors with clear obstacle labels and distinct colors.

        Args:
            anchors: Dictionary of anchor rules from explain()
            max_anchors: Maximum number of anchors to visualize
        """
        if not anchors:
            return None

        top_anchors = list(anchors.items())[:max_anchors]
        fig, axes = plt.subplots(len(top_anchors), 1, figsize=(10, 4 * len(top_anchors)))

        if len(top_anchors) == 1:
            axes = [axes]

        cmap = colormaps['tab10']
        obstacle_keys = list(self.env.obstacle_shapes.keys())

        for i, (rule_str, anchor_data) in enumerate(top_anchors):
            ax = axes[i]
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            # ax.set_ylim(-0.5, self.grid_size - 0.5)  # Fixed: Don't invert y-axis
            ax.set_ylim(self.grid_size - 0.5, -0.5)  # Invert y axis for correct orientation
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(0, self.grid_size, 1))
            ax.set_yticks(np.arange(0, self.grid_size, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

            rule = anchor_data["rule"]

            for idx, shape_id in enumerate(obstacle_keys):
                if idx < len(rule) and rule[idx] is not None:
                    action = rule[idx]  # 1 = keep, 0 = remove
                    color = cmap(idx % 10)
                    shape = self.env.obstacle_shapes[shape_id]

                    # Draw each cell in the shape
                    for pos in shape:
                        x, y = pos
                        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                            ax.add_patch(plt.Rectangle(
                                (y - 0.5, x - 0.5), 1, 1,
                                facecolor=color,
                                alpha=0.6 if action == 1 else 0.9,
                                edgecolor='black',
                                linewidth=1.0,
                                hatch='//' if action == 0 else None
                            ))

                    # Annotate the first cell with ID and action
                    fx, fy = shape[0]
                    ax.text(fy, fx, f"#{shape_id}\n{'Keep' if action == 1 else 'Remove'}",
                            ha='center', va='center', fontsize=8,
                            color='black', weight='bold',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

            # Mark agent and goal positions
            if self.env.agent_pos:
                ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0],
                        color='blue', s=150, marker='o', label='Start')
            if self.env.goal_pos:
                ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0],
                        color='green', s=150, marker='*', label='Goal')

            ax.set_title(f"Anchor #{i+1}: {anchor_data['description']}", fontsize=10)

            if i == 0:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

        plt.tight_layout()
        return fig