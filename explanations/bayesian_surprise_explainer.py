import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

class BayesianSurpriseExplainer:
    def __init__(self):
        self.environment = None
        self.planner = None

    def set_environment(self, env, planner):
        self.environment = env
        self.planner = planner

    def compute_surprise(self, prior_probs, posterior_probs):
        return entropy(posterior_probs, prior_probs)

    def explain(self, perturbation_mode='remove', num_samples=30):
        prior = np.ones(num_samples) / num_samples
        posterior = []
        for i in range(num_samples):
            perturbed_env = self.environment.sample_perturbation(mode=perturbation_mode)
            self.planner.set_environment(perturbed_env.agent_pos,
                                         perturbed_env.goal_pos,
                                         perturbed_env.grid_size,
                                         perturbed_env.obstacles)
            path = self.planner.plan()
            score = 1.0 if path else 0.0
            posterior.append(score)
        posterior = np.array(posterior)
        if posterior.sum() == 0:
            return 0.0  # Avoid division by zero
        posterior = posterior / posterior.sum()
        return self.compute_surprise(prior, posterior)

    def visualize(self, posterior_probs):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(range(len(posterior_probs)), posterior_probs, color='orange')
        ax.set_title("Posterior Probability Distribution (Perturbed Outcomes)")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Probability")
        return fig
