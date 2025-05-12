import numpy as np
import math
import matplotlib.pyplot as plt

class WoEExplainer:
    def __init__(self):
        self.env = None
        self.planner = None

    def set_environment(self, env, planner):
        self.env = env
        self.planner = planner

    def compute_posterior(self, success_rate):
        """Convert success rate into a probability-like posterior."""
        epsilon = 1e-6
        return np.clip(success_rate, epsilon, 1 - epsilon)

    def compute_woe(self, p_g, p_g_prime):
        """Compute Weight of Evidence (WoE) between two goal hypotheses."""
        return math.log(p_g / p_g_prime)

    def rank_observations_by_woe(self, posterior_map):
        """
        Given a mapping of observations to posterior values for (g, g'),
        return ranked list of (obs, woe).
        """
        woe_list = []
        for obs, (p_g, p_g_prime) in posterior_map.items():
            woe_val = self.compute_woe(p_g, p_g_prime)
            woe_list.append((obs, woe_val))
        woe_list.sort(key=lambda x: -x[1])  # Descending
        return woe_list

    def explain(self, posterior_map):
        """
        Run WoE ranking over provided posterior_map.
        Returns top positive and negative WoE markers.
        """
        ranked = self.rank_observations_by_woe(posterior_map)
        return {
            "ranked": ranked,
            "top_positive": ranked[0] if ranked else None,
            "top_negative": min(ranked, key=lambda x: x[1]) if ranked else None
        }

    def visualize(self, ranked_woe, top_k=10):
        """
        Simple bar plot of WoE values for top-k observations.
        """
        if not ranked_woe:
            return None

        top = ranked_woe[:top_k]
        obs_labels = [str(o[0]) for o in top]
        woe_vals = [o[1] for o in top]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(obs_labels, woe_vals, color='skyblue')
        ax.set_ylabel("Weight of Evidence (WoE)")
        ax.set_title("Top WoE Observations (Goal g vs. g')")
        ax.axhline(0, color='black', linewidth=1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
