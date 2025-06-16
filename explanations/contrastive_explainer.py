import numpy as np
import matplotlib.pyplot as plt
import itertools

class ContrastiveExplainer:
    """
    Contrastive explanation: Why trajectory A (factual) rather than B (expected)?
    Compares two planning outcomes and highlights obstacles responsible for the difference.
    """

    def __init__(self):
        self.env = None
        self.planner = None
        self.grid_size = None

    def set_environment(self, env, planner):
        self.env = env
        self.planner = planner
        self.grid_size = env.grid_size

    def is_close(self, shape, path_set, threshold=0):
        return any(
            abs(x - px) + abs(y - py) <= threshold
            for (x, y) in shape
            for (px, py) in path_set
        )

    def explain(self, factual_path, contrastive_path, minimal=False, proximity_threshold=0):
        if not factual_path or not contrastive_path:
            raise ValueError("Both factual_path and contrastive_path must be provided")

        fp_set = set(tuple(p) for p in factual_path)
        cp_set = set(tuple(p) for p in contrastive_path)

        contrastive = {
            "factual_path": factual_path,
            "alt_path": contrastive_path,
            "factual_length": len(factual_path),
            "alt_length": len(contrastive_path),
            "obstacles_affecting_choice": []
        }

        # Initial full set of differing obstacles
        candidates = []
        for shape_id, shape in self.env.obstacle_shapes.items():
            close_to_factual = self.is_close(shape, fp_set, threshold=proximity_threshold)
            close_to_alt = self.is_close(shape, cp_set, threshold=proximity_threshold)

            if close_to_factual != close_to_alt:
                candidates.append({
                    "id": shape_id,
                    "near_factual": close_to_factual,
                    "near_alt": close_to_alt,
                    "shape": shape
                })

        if minimal:
            candidates = self.compute_minimal_contrastive_set(candidates, factual_path, contrastive_path)

        contrastive["obstacles_affecting_choice"] = candidates
        return contrastive

    def compute_minimal_contrastive_set(self, candidates, factual_path, contrastive_path):
        result = []
        for c in candidates:
            modified_env = self.env.clone()
            modified_env.remove_obstacle_shape(c["id"])

            self.planner.set_environment(
                start=modified_env.agent_pos,
                goal=modified_env.goal_pos,
                grid_size=modified_env.grid_size,
                obstacles=modified_env.obstacles
            )
            new_path = self.planner.plan()

            if new_path and new_path != factual_path and new_path == contrastive_path:
                result.append(c)
        return result

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

        if contrastive_result["factual_path"]:
            xs, ys = zip(*contrastive_result["factual_path"])
            ax.plot(ys, xs, color='orange', linewidth=3, label='Factual Path')

        if contrastive_result["alt_path"]:
            xs, ys = zip(*contrastive_result["alt_path"])
            ax.plot(ys, xs, color='purple', linewidth=3, label='Expected Path')

        if self.env.agent_pos:
            ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0], color='blue', s=150, marker='o', label='Start')
        if self.env.goal_pos:
            ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0], color='green', s=150, marker='*', label='Goal')

        ax.set_title("Contrastive Explanation: Why A not B?")
        ax.legend(loc='upper right')
        return fig
