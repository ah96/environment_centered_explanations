
# Explainer: Responsibility and Blame
# Measures minimal number of changes needed to flip the outcome (based on Halpern & Pearl)

import matplotlib.pyplot as plt

class ResponsibilityExplainer:
    def __init__(self):
        self.environment = None
        self.planner = None

    def set_environment(self, env, planner):
        self.environment = env
        self.planner = planner

    def explain(self, max_changes=2):
        base_path = self.planner.plan()
        if not base_path:
            return {}

        responsible_obstacles = []
        for obs in self.environment.obstacles:
            for i in range(1, max_changes + 1):
                new_obstacles = [o for o in self.environment.obstacles if o != obs]
                self.planner.set_environment(self.environment.agent_pos,
                                             self.environment.goal_pos,
                                             self.environment.grid_size,
                                             new_obstacles)
                path = self.planner.plan()
                if not path:
                    responsible_obstacles.append((obs, i))
                    break
        return {"responsible_obstacles": responsible_obstacles}

    def visualize(self, result):
        if not result or "responsible_obstacles" not in result:
            return None

        resp_obs = [r[0] for r in result["responsible_obstacles"]]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, self.environment.grid_size - 0.5)
        ax.set_ylim(self.environment.grid_size - 0.5, -0.5)
        ax.grid(True)

        for shape in self.environment.obstacle_shapes.values():
            for x, y in shape:
                color = 'red' if [x, y] in resp_obs else 'gray'
                alpha = 1.0 if [x, y] in resp_obs else 0.3
                ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color=color, alpha=alpha))

        ax.set_title(f"Responsible Obstacles: {len(resp_obs)}")
        return fig
