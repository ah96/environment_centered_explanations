
import copy
import matplotlib.pyplot as plt
import numpy as np

class COSEExplainer:
    def __init__(self):
        self.env = None
        self.planner = None
        self.grid_size = None

    def set_environment(self, env, planner):
        self.env = env.clone()
        self.planner = planner
        self.grid_size = env.grid_size

    def explain(self, max_steps=None, affordance_mode="remove", use_guided_ranking=False, guidance_scores=None, callback=None):
        """
        COSE: Critical Obstacle Set Explainer

        Args:
            max_steps (int): Optional limit to number of obstacle removals
            affordance_mode (str): "remove", "move", etc.
            use_guided_ranking (bool): Whether to use guidance_scores for ranking
            guidance_scores (dict): Dictionary {shape_id: score}
            callback (func): Optional progress callback

        Returns:
            critical_set (list): List of obstacle IDs whose removal or modification restores feasibility
        """
        state = self.env.get_state()
        all_obstacles = list(state["obstacle_shapes"].keys())
        if not all_obstacles:
            return []

        if use_guided_ranking and guidance_scores:
            # Sort obstacles by decreasing guidance score (higher = more important)
            sorted_obstacles = sorted(all_obstacles, key=lambda o: -abs(guidance_scores.get(o, 0)))
        else:
            import random
            sorted_obstacles = all_obstacles.copy()
            random.shuffle(sorted_obstacles)

        removed_obstacles = []
        original_state = {
            'obstacles': copy.deepcopy(self.env.obstacles),
            'obstacle_shapes': copy.deepcopy(self.env.obstacle_shapes)
        }

        for i, shape_id in enumerate(sorted_obstacles):
            if callback:
                callback(i, len(sorted_obstacles))

            # Apply perturbation using the selected affordance mode
            self.env.generate_perturbation(strategy="custom", shape_id=shape_id, mode=affordance_mode)
            removed_obstacles.append(shape_id)

            # Re-run planner
            planner_class = type(self.planner)
            planner = planner_class()
            planner.set_environment(
                start=self.env.agent_pos,
                goal=self.env.goal_pos,
                grid_size=self.env.grid_size,
                obstacles=self.env.obstacles
            )
            path = planner.plan()
            if path:
                break  # Stop as soon as a path is found
            if max_steps is not None and len(removed_obstacles) >= max_steps:
                break

        # Restore environment for visualization or further use
        self.env.obstacles = original_state['obstacles']
        self.env.obstacle_shapes = original_state['obstacle_shapes']
        return removed_obstacles

    def visualize(self, removed_obstacles):
        """Visualize the obstacles removed or modified in the critical set"""
        grid = np.zeros((self.grid_size, self.grid_size))

        for shape_id in removed_obstacles:
            if shape_id in self.env.obstacle_shapes:
                for x, y in self.env.obstacle_shapes[shape_id]:
                    grid[x, y] = 1

        fig, ax = plt.subplots(figsize=(8, 8))
        heatmap = ax.imshow(grid, cmap='Reds', interpolation='nearest')
        ax.set_title("COSE: Critical Obstacles to Modify for Success")
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.grid(True)
        plt.colorbar(heatmap, ax=ax, label="Obstacle Modification Indicator")
        return fig
