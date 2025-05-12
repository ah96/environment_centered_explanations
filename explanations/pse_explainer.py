import matplotlib.pyplot as plt
import itertools

class PSEExplainer:
    def __init__(self):
        self.environment = None
        self.planner = None

    def set_environment(self, env, planner):
        self.environment = env
        self.planner = planner

    def explain(self, threshold=0.9):
        original_path = self.planner.plan()
        if not original_path:
            return {}

        for r in range(1, len(self.environment.obstacles) + 1):
            for subset in itertools.combinations(self.environment.obstacles, r):
                reduced_env = self.environment.copy()
                reduced_env.obstacles = list(subset)
                self.planner.set_environment(reduced_env.agent_pos,
                                             reduced_env.goal_pos,
                                             reduced_env.grid_size,
                                             reduced_env.obstacles)
                path = self.planner.plan()
                if path and len(path) / len(original_path) >= threshold:
                    return {"sufficient_obstacles": subset}
        return {"sufficient_obstacles": []}

    def visualize(self, result):
        if not result or "sufficient_obstacles" not in result:
            return None

        subset = result["sufficient_obstacles"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, self.environment.grid_size - 0.5)
        ax.set_ylim(self.environment.grid_size - 0.5, -0.5)
        ax.grid(True)

        for shape in self.environment.obstacle_shapes.values():
            for x, y in shape:
                color = 'red' if [x, y] in subset else 'gray'
                alpha = 1.0 if [x, y] in subset else 0.3
                ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color=color, alpha=alpha))

        ax.set_title(f"PSE: {len(subset)} sufficient obstacle(s)")
        return fig
