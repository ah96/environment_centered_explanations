import numpy as np
import time
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error

def compute_path_metrics(original_path, new_path, original_time, new_time, grid_size):
    original_length = len(original_path) if original_path else 0
    new_length = len(new_path) if new_path else 0
    path_restored = original_length == 0 and new_length > 0

    if original_length > 0:
        path_length_diff = new_length - original_length
        path_length_change_pct = (path_length_diff / original_length) * 100
    else:
        penalty = grid_size * grid_size + 1
        path_length_change_pct = 100 * (1 - (new_length / penalty)) if new_length else 0
        path_length_diff = new_length

    exec_time_diff = new_time - original_time
    path_success = new_length > 0

    return {
        "original_length": original_length,
        "new_length": new_length,
        "path_restored": path_restored,
        "path_length_diff": path_length_diff,
        "path_length_change_pct": path_length_change_pct,
        "exec_time_diff": exec_time_diff,
        "path_success": path_success,
    }

def compute_faithfulness(original_scores, perturbed_scores):
    """Assumes scores are relevance maps or explanation masks"""
    return 1 - mean_squared_error(original_scores, perturbed_scores)

def compute_consistency(explanations_list):
    """Computes average pairwise cosine similarity between explanation vectors"""
    sims = []
    for i in range(len(explanations_list)):
        for j in range(i+1, len(explanations_list)):
            sims.append(1 - cosine(explanations_list[i], explanations_list[j]))
    return np.mean(sims) if sims else 0

def compute_planner_robustness(original_plan, perturbed_plans):
    """Ratio of perturbed plans that reach the goal vs. total"""
    success_count = sum([len(p) > 0 for p in perturbed_plans])
    return success_count / len(perturbed_plans) if perturbed_plans else 0

def compute_runtime_statistics(runtime_list):
    """Basic runtime metrics"""
    return {
        "mean_runtime": np.mean(runtime_list),
        "std_runtime": np.std(runtime_list),
        "max_runtime": np.max(runtime_list),
        "min_runtime": np.min(runtime_list),
    }

def compute_explanation_overlap(expl1, expl2):
    """Assumes binary explanation maps of same shape"""
    expl1, expl2 = np.array(expl1), np.array(expl2)
    return np.sum(np.logical_and(expl1, expl2)) / np.sum(np.logical_or(expl1, expl2))

def compute_explanation_sparsity(explanation):
    """Ratio of non-zero entries to total size"""
    explanation = np.array(explanation)
    return 1.0 - (np.count_nonzero(explanation) / explanation.size)

def compute_explanation_sensitivity(expl_original, expl_perturbed_list):
    """Mean distance of perturbed explanations to original"""
    diffs = [np.linalg.norm(np.array(expl_original) - np.array(e)) for e in expl_perturbed_list]
    return np.mean(diffs)

def compute_explanation_redundancy(explanation):
    """Variance of the explanation values"""
    explanation = np.array(explanation)
    return np.var(explanation)

