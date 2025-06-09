import numpy as np
import math
import matplotlib.pyplot as plt

class WoEExplainer:
    def __init__(self):
        self.env = None
        self.planner = None
        self.grid_size = None

    def set_environment(self, env, planner):
        self.env = env
        self.planner = planner
        self.grid_size = env.grid_size

    def compute_posterior(self, success_rate):
        epsilon = 1e-6
        return np.clip(success_rate, epsilon, 1 - epsilon)

    def compute_woe(self, p_g, p_g_prime):
        return math.log(p_g / p_g_prime)

    def explain(self, factual_path, contrastive_path, trials=10):
        if not factual_path or not contrastive_path:
            raise ValueError("Both factual_path and contrastive_path must be provided")

        # Store original environment
        fp_set = set(tuple(p) for p in factual_path)
        cp_set = set(tuple(p) for p in contrastive_path)

        posterior_map = {}

        for shape_id in self.env.obstacle_shapes:
            success_A = 0
            success_B = 0

            for _ in range(trials):
                mod_env = self.env.clone()
                mod_env.remove_obstacle_shape(shape_id)

                self.planner.set_environment(
                    start=mod_env.agent_pos,
                    goal=mod_env.goal_pos,
                    grid_size=mod_env.grid_size,
                    obstacles=mod_env.obstacles
                )

                new_path = self.planner.plan()

                if not new_path:
                    continue

                new_path_set = set(tuple(p) for p in new_path)

                if new_path_set == fp_set:
                    success_A += 1
                elif new_path_set == cp_set:
                    success_B += 1

            p_A = self.compute_posterior(success_A / trials)
            p_B = self.compute_posterior(success_B / trials)
            posterior_map[shape_id] = (p_A, p_B)

        ranked = self.rank_observations_by_woe(posterior_map)

        return {
            "ranked": ranked,
            "posterior_map": posterior_map,
            "top_positive": ranked[0] if ranked else None,
            "top_negative": min(ranked, key=lambda x: x[1]) if ranked else None,
            "factual_path": factual_path,
            "alt_path": contrastive_path
        }

    def rank_observations_by_woe(self, posterior_map):
        woe_list = []
        for obs, (p_g, p_g_prime) in posterior_map.items():
            woe_val = self.compute_woe(p_g, p_g_prime)
            woe_list.append((obs, woe_val))
        woe_list.sort(key=lambda x: -x[1])
        return woe_list

    def visualize(self, result, top_k=10):
        if not result:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, self.grid_size, 1))
        ax.set_yticks(np.arange(0, self.grid_size, 1))
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

        # Draw obstacles with WoE coloring
        cmap = plt.cm.RdYlBu
        max_woe = max(abs(w) for _, w in result["ranked"][:top_k]) if result["ranked"] else 1

        for obs_id, shape in self.env.obstacle_shapes.items():
            woe_val = dict(result["ranked"]).get(obs_id, 0)
            norm_val = 0.5 + 0.5 * (woe_val / max_woe) if max_woe != 0 else 0.5
            color = cmap(norm_val)

            for x, y in shape:
                ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1,
                                           facecolor=color,
                                           edgecolor='black',
                                           linewidth=1.5,
                                           alpha=0.7))

            if shape:
                fx, fy = shape[0]
                ax.text(fy, fx, f"#{obs_id}", ha='center', va='center',
                        fontsize=8, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Draw paths
        if result["factual_path"]:
            xs, ys = zip(*result["factual_path"])
            ax.plot(ys, xs, color='orange', linewidth=3, label='Factual Path')

        if result["alt_path"]:
            xs, ys = zip(*result["alt_path"])
            ax.plot(ys, xs, color='purple', linewidth=3, label='Expected Path')

        ax.legend(loc='upper right')
        ax.set_title("WoE Explanation: Obstacles influencing A vs. B")
        return fig
