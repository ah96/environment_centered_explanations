import numpy as np
import matplotlib.pyplot as plt
import itertools

class ContrastiveExplainer:
    """
    Contrastive explanation: Why trajectory A (factual) rather than B (expected)?
    Compares two planning outcomes and highlights obstacles responsible for the difference.
    """

    def __init__(self, env=None, alt_env=None, factual_path=None, contrastive_path=None):
        """
        Initialize with optional environments and paths.

        Args:
            env (GridWorldEnv): Original environment (for path A).
            alt_env (GridWorldEnv): Contrastive environment (for path B).
            factual_path (List[Tuple[int, int]]): Path A.
            contrastive_path (List[Tuple[int, int]]): Path B.
        """
        self.env = env
        self.alt_env = alt_env
        self.factual_path = factual_path
        self.contrastive_path = contrastive_path
        self.planner = None
        self.grid_size = env.grid_size if env else None

    def set_environment(self, env, planner):
        """
        Set the environment and planner for the explainer.

        Args:
            env: GridWorldEnv instance
            planner: Planner instance with set_environment and plan methods
        """
        self.env = env
        self.planner = planner
        self.grid_size = env.grid_size

    def is_close(self, shape, path_set, threshold=0):
        """
        Checks if any cell in a given obstacle shape is within a given
        Manhattan distance (threshold) to any point on the given path.

        Args:
            shape: list of (x, y) obstacle coordinates
            path_set: set of (x, y) path coordinates
            threshold: maximum distance to consider "close"

        Returns:
            True if any obstacle point is close to the path
        """
        return any(
            abs(x - px) + abs(y - py) <= threshold
            for (x, y) in shape
            for (px, py) in path_set
        )

    def explain(self, factual_path, contrastive_path, minimal=True, proximity_threshold=0, perturbation_mode="remove"):
        """
        Generate a contrastive explanation by identifying obstacles near
        either the factual or contrastive path, but not both.

        Args:
            factual_path: path actually followed (Plan A)
            contrastive_path: expected or alternative path (Plan B)
            minimal: if True, filters obstacles down to minimal sufficient subset
            proximity_threshold: distance from path within which obstacles are considered
            perturbation_mode: perturbation type ('remove' or 'move')

        Returns:
            A dictionary with explanation data including affected obstacles
        """
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

        # Identify obstacle shapes that are close to one path but not the other
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
            candidates = self.compute_minimal_contrastive_set(
                candidates, factual_path, contrastive_path, perturbation_mode=perturbation_mode
            )

        contrastive["obstacles_affecting_choice"] = candidates
        return contrastive

    def compute_minimal_contrastive_set(self, candidates, factual_path, contrastive_path, perturbation_mode="remove"):
        """
        Identifies minimal subsets of obstacles (including joint combinations) that,
        when perturbed (removed or moved), cause the planner to change its output
        from the factual path to the contrastive path.

        Args:
            candidates: List of obstacle metadata dictionaries
            factual_path: The path actually taken
            contrastive_path: The contrastive/alternative path
            perturbation_mode: Type of perturbation ('remove' or 'move')

        Returns:
            List of obstacle metadata dictionaries that minimally cause the path shift
        """
        minimal_subsets = []
        n = len(candidates)

        for r in range(1, n + 1):
            found = False
            for subset in itertools.combinations(candidates, r):
                modified_env = self.env.clone()

                # Apply perturbations
                for c in subset:
                    if perturbation_mode == "remove":
                        modified_env.remove_obstacle_shape(c["id"])
                    elif perturbation_mode == "move":
                        modified_env.move_obstacle_shape(c["id"])
                    else:
                        raise ValueError("Unsupported perturbation mode")

                self.planner.set_environment(
                    start=modified_env.agent_pos,
                    goal=modified_env.goal_pos,
                    grid_size=modified_env.grid_size,
                    obstacles=modified_env.obstacles
                )
                new_path = self.planner.plan()

                if new_path and new_path != factual_path and new_path == contrastive_path:
                    minimal_subsets.append(list(subset))
                    found = True

            if found:
                break

        flat_set = {c["id"]: c for subset in minimal_subsets for c in subset}
        return list(flat_set.values())

    def visualize(self, contrastive_result):
        """
        Visualizes the factual and contrastive paths along with obstacle shapes
        and highlights the differences that affected the planning decision.

        Args:
            contrastive_result: result dictionary from the explain() method

        Returns:
            A matplotlib figure object with the visual explanation
        """
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
