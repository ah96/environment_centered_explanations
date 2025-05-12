import numpy as np
import matplotlib.pyplot as plt

class GoalCounterfactualExplainer:
    def __init__(self):
        self.env = None
        self.planner = None
        self.grid_size = None

    def set_environment(self, env, planner):
        self.env = env
        self.planner = planner
        self.grid_size = env.grid_size

    def explain(self, max_goals=5):
        # Factual path
        self.planner.set_environment(
            start=self.env.agent_pos,
            goal=self.env.goal_pos,
            grid_size=self.env.grid_size,
            obstacles=self.env.obstacles
        )
        factual_path = self.planner.plan()
        factual_length = len(factual_path) if factual_path else float('inf')

        # Generate counterfactual goals
        counterfactuals = []
        tried = 0
        added = 0
        max_attempts = 100

        while added < max_goals and tried < max_attempts:
            new_goal = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if new_goal == self.env.goal_pos or new_goal in self.env.obstacles:
                tried += 1
                continue

            self.planner.set_environment(
                start=self.env.agent_pos,
                goal=new_goal,
                grid_size=self.env.grid_size,
                obstacles=self.env.obstacles
            )
            cf_path = self.planner.plan()
            cf_length = len(cf_path) if cf_path else float('inf')

            if cf_path and cf_length != factual_length:
                counterfactuals.append({
                    "goal": new_goal,
                    "path": cf_path,
                    "length": cf_length,
                    "delta": cf_length - factual_length
                })
                added += 1
            tried += 1

        return counterfactuals

    def visualize(self, counterfactuals, top_k=3):
        if not counterfactuals:
            return None

        top_k = min(top_k, len(counterfactuals))
        fig, axes = plt.subplots(top_k, 1, figsize=(8, 4 * top_k))

        if top_k == 1:
            axes = [axes]

        for i in range(top_k):
            cf = counterfactuals[i]
            ax = axes[i]
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(self.grid_size - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.grid(True)

            for shape in self.env.obstacle_shapes.values():
                for x, y in shape:
                    ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='gray'))

            ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0], color='blue', s=120, marker='o', label='Start')
            ax.scatter(cf["goal"][1], cf["goal"][0], color='red', s=120, marker='*', label='CF Goal')

            path = cf["path"]
            if path:
                path_x = [p[1] for p in path]
                path_y = [p[0] for p in path]
                ax.plot(path_x, path_y, color='orange', linewidth=2)

            ax.set_title(f"Goal CF #{i+1}: Goal = {cf['goal']}, Length = {cf['length']} (Δ = {cf['delta']:+})")
            ax.legend()

        plt.tight_layout()
        return fig
