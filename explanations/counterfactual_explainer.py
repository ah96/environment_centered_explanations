import numpy as np
import matplotlib.pyplot as plt
import itertools

class CounterfactualExplainer:
    """
    Counterfactual explanation: What is the minimal set of obstacles whose removal
    would significantly improve the path (e.g., shorten it)?
    """

    def __init__(self):
        self.env = None
        self.planner = None
        self.grid_size = None

    def set_environment(self, env, planner):
        self.env = env
        self.planner = planner
        self.grid_size = env.grid_size

    def explain(self, max_subset_size=2, improvement_threshold=1, perturbation_mode="remove"):
        obstacle_keys = list(self.env.obstacle_shapes.keys())
        num_obstacles = len(obstacle_keys)
        if num_obstacles == 0:
            return []

        baseline_path = self.planner.plan()
        baseline_length = len(baseline_path) if baseline_path else float('inf')
        counterfactuals = []

        original_obstacles = self.env.obstacles.copy()

        for subset_size in range(1, max_subset_size + 1):
            for subset in itertools.combinations(range(num_obstacles), subset_size):
                combination = [1] * num_obstacles
                for i in subset:
                    combination[i] = 0  # remove obstacle

                original_state, _ = self.env.generate_perturbation(combination=combination, mode=perturbation_mode)
                path = self.planner.plan(
                    self.env.agent_pos,
                    self.env.goal_pos,
                    self.env.obstacles
                )
                new_length = len(path) if path else float('inf')
                self.env.restore_from_perturbation(original_state)

                if new_length + improvement_threshold < baseline_length:
                    counterfactuals.append({
                        "obstacles_removed": subset,
                        "new_length": new_length,
                        "improvement": baseline_length - new_length
                    })

        self.env.obstacles = original_obstacles.copy()
        return sorted(counterfactuals, key=lambda x: x["improvement"], reverse=True)

    def visualize(self, counterfactuals, top_k=3):
        if not counterfactuals:
            return None

        top_counterfactuals = counterfactuals[:top_k]
        fig, axes = plt.subplots(len(top_counterfactuals), 1, figsize=(10, 4 * len(top_counterfactuals)))

        if len(top_counterfactuals) == 1:
            axes = [axes]

        cmap = plt.cm.tab10

        for i, cf in enumerate(top_counterfactuals):
            ax = axes[i]
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(self.grid_size - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(0, self.grid_size, 1))
            ax.set_yticks(np.arange(0, self.grid_size, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

            for idx, shape_id in enumerate(self.env.obstacle_shapes):
                color = cmap(idx % 10)
                shape = self.env.obstacle_shapes[shape_id]
                for x, y in shape:
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        facecolor = color
                        alpha = 1.0 if idx not in cf["obstacles_removed"] else 0.2
                        ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1,
                                                   facecolor=facecolor, alpha=alpha,
                                                   edgecolor='black', linewidth=1))

                if shape:
                    fx, fy = shape[0]
                    label = f"#{shape_id}"
                    if idx in cf["obstacles_removed"]:
                        label += " (removed)"
                    ax.text(fy, fx, label, ha='center', va='center',
                            fontsize=8, color='black',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            if self.env.agent_pos:
                ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0], color='blue', s=150, marker='o', label='Start')
            if self.env.goal_pos:
                ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0], color='green', s=150, marker='*', label='Goal')

            ax.set_title(f"Counterfactual #{i+1}: Remove {cf['obstacles_removed']} → path length = {cf['new_length']} (Δ = {cf['improvement']})")
            ax.legend(loc='upper right')

        plt.tight_layout()
        return fig