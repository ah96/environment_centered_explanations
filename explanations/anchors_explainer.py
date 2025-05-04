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

        candidate_anchors = []

        # Basic one-obstacle rules
        for i in range(num_obstacles):
            keep_rule = [None] * num_obstacles
            keep_rule[i] = 1
            candidate_anchors.append(keep_rule)

            remove_rule = [None] * num_obstacles
            remove_rule[i] = 0
            candidate_anchors.append(remove_rule)

        # Random multi-obstacle rules
        for _ in range(min(20, num_obstacles * 2)):
            rule_size = random.randint(2, min(3, num_obstacles))
            rule = [None] * num_obstacles
            for _ in range(rule_size):
                idx = random.randint(0, num_obstacles - 1)
                rule[idx] = random.choice([0, 1])
            candidate_anchors.append(rule)

        anchor_results = {}
        original_obstacles = self.env.obstacles.copy()
        total_iterations = len(candidate_anchors) * num_samples
        iteration = 0

        for anchor_idx, anchor_rule in enumerate(candidate_anchors):
            if callback:
                callback(anchor_idx, len(candidate_anchors))

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

                # match user-desired outcome type
                if majority_outcome == detect_changes and precision >= precision_threshold and coverage >= min_coverage:
                    rule_description = []
                    for i, value in enumerate(anchor_rule):
                        if value is not None:
                            action = "keep" if value == 1 else "remove"
                            rule_description.append(f"{action} obstacle #{i}")

                    rule_str = " AND ".join(rule_description)
                    outcome_str = "path changes significantly" if detect_changes else "path remains similar"

                    anchor_results[rule_str] = {
                        "rule": anchor_rule,
                        "precision": precision,
                        "coverage": coverage,
                        "outcome": majority_outcome,
                        "description": f"When we {rule_str}, the {outcome_str} (precision: {precision:.2f})"
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
            ax.set_ylim(-0.5, self.grid_size - 0.5)
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