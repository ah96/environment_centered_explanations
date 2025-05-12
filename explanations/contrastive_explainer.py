import numpy as np
import matplotlib.pyplot as plt

class ContrastiveExplainer:
    """
    Contrastive explanation: Why was one path chosen over another?
    Compares two planning outcomes and highlights obstacles responsible for preferring one.
    """

    def __init__(self):
        self.env = None
        self.planner = None
        self.grid_size = None

    def set_environment(self, env, planner):
        self.env = env
        self.planner = planner
        self.grid_size = env.grid_size

    def explain(self, alternative_combination=None, perturbation_mode="remove"):
        # Plan A: factual plan (original environment)
        factual_path = self.planner.plan()
        factual_length = len(factual_path) if factual_path else float('inf')
        print("[DEBUG] factual_path:", factual_path)

        # Plan B: contrastive plan (e.g., fewer obstacles)
        obstacle_keys = list(self.env.obstacle_shapes.keys())
        num_obstacles = len(obstacle_keys)

        if alternative_combination is None:
            alternative_combination = [1] * num_obstacles
            # Try removing obstacles near factual path
            factual_set = set(tuple(p) for p in factual_path)
            for i, (key, shape) in enumerate(self.env.obstacle_shapes.items()):
                if any(tuple(p) in factual_set for p in shape):
                    alternative_combination[i] = 0

        original_state, _ = self.env.generate_perturbation(combination=alternative_combination, mode=perturbation_mode)

        alt_path = self.planner.plan()
        alt_length = len(alt_path) if alt_path else float('inf')
        print("[DEBUG] alt_path:", alt_path)

        self.env.restore_from_perturbation(original_state)

        # Initialize result
        contrastive = {
            "factual_path": factual_path,
            "alt_path": alt_path,
            "factual_length": factual_length,
            "alt_length": alt_length,
            "obstacles_affecting_choice": []
        }

        if factual_path and alt_path:
            fp_set = set(tuple(p) for p in factual_path)
            ap_set = set(tuple(p) for p in alt_path)

            for shape_id, shape in self.env.obstacle_shapes.items():
                close_to_factual = any(tuple(p) in fp_set for p in shape)
                close_to_alt = any(tuple(p) in ap_set for p in shape)
                if close_to_factual != close_to_alt:
                    contrastive["obstacles_affecting_choice"].append({
                        "id": shape_id,
                        "near_factual": close_to_factual,
                        "near_alt": close_to_alt
                    })

        return contrastive

    def visualize(self, contrastive_result):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, self.grid_size, 1))
        ax.set_yticks(np.arange(0, self.grid_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

        # Color code obstacles
        cmap = plt.cm.tab10
        for shape_id, shape in self.env.obstacle_shapes.items():
            color = cmap(shape_id % 10)
            alpha = 0.3
            edge = 'black'

            labels = [o for o in contrastive_result["obstacles_affecting_choice"] if o["id"] == shape_id]
            if labels:
                label = labels[0]
                alpha = 0.8
                edge = 'red' if label["near_factual"] else 'blue'

            for x, y in shape:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1,
                                               facecolor=color,
                                               edgecolor=edge,
                                               linewidth=1.5,
                                               alpha=alpha))

            if shape:
                fx, fy = shape[0]
                ax.text(fy, fx, f"#{shape_id}", ha='center', va='center',
                        fontsize=8, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Draw paths
        drew_path = False
        if contrastive_result["factual_path"]:
            xs, ys = zip(*contrastive_result["factual_path"])
            ax.plot(ys, xs, color='orange', linewidth=3, label='Factual Path')
            drew_path = True

        if contrastive_result["alt_path"]:
            xs, ys = zip(*contrastive_result["alt_path"])
            ax.plot(ys, xs, color='purple', linewidth=3, label='Alternative Path')
            drew_path = True

        # Draw start/goal
        if self.env.agent_pos:
            ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0],
                       color='blue', s=150, marker='o', label='Start')
        if self.env.goal_pos:
            ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0],
                       color='green', s=150, marker='*', label='Goal')

        # Adjust title based on availability
        title = "Contrastive Explanation:"
        if not contrastive_result["factual_path"]:
            title += "\n(No factual path found)"
        elif not contrastive_result["alt_path"]:
            title += "\n(No alternative path found)"
        else:
            title += "\nWhy this path vs. that one?"

        ax.set_title(title)
        if drew_path:
            ax.legend(loc='upper right')
        return fig