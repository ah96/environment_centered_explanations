import numpy as np
import heapq
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import copy
from sklearn.metrics import pairwise_distances
from functools import partial
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
import scipy as sp

#from . import lime_image

# GridWorld environment for robot navigation
default_grid_size = 10

class GridWorldEnv:
    def __init__(self, grid_size=default_grid_size, num_obstacles=5):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        self.obstacles = self.generate_obstacles()
        return self.get_state()

    def generate_obstacles(self):
        # Randomly place obstacles in the grid avoiding start and goal
        obs = []
        while len(obs) < self.num_obstacles:
            pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if pos != self.agent_pos and pos != self.goal_pos and pos not in obs:
                obs.append(pos)
        return obs

    def get_state(self):
        return {
            "grid_size": self.grid_size,
            "agent": self.agent_pos,
            "goal": self.goal_pos,
            "obstacles": self.obstacles
        }

# A* planner for pathfinding
def a_star(grid_size, start, goal, obstacles):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {tuple(start): 0}

    def h(pos):
        # Manhattan distance heuristic
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct path
            path = []
            while tuple(current) in came_from:
                path.append(current)
                current = came_from[tuple(current)]
            path.append(start)
            return path[::-1]
        # Explore neighbors
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = [current[0] + dx, current[1] + dy]
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and neighbor not in obstacles:
                tentative_g = g_score[tuple(current)] + 1
                if tuple(neighbor) not in g_score or tentative_g < g_score[tuple(neighbor)]:
                    came_from[tuple(neighbor)] = current
                    g_score[tuple(neighbor)] = tentative_g
                    heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
    return None

# Replanning-based explanation to detect critical obstacles
# So far, each obstacle is removed once and the path is replanned
# If the path length decreases, the obstacle is considered critical
def replanning_explanation(start, goal, obstacles, grid_size):
    paths = []
    obstacles_ = [obstacles]
    
    base_path = a_star(grid_size, start, goal, obstacles)
    if base_path == None:
        print("\nNo base path found.")
        paths.append([-1])
    else:
        paths.append(base_path)
    
    explanations = []
    for obs in obstacles:
        reduced_obs = [o for o in obstacles if o != obs]
        obstacles_.append(reduced_obs)
        new_path = a_star(grid_size, start, goal, reduced_obs)
        if new_path: 
            if base_path == None:
                explanations.append(f"Obstacle at {obs} caused path to be blocked.")
            elif new_path and len(new_path) < len(base_path):
                explanations.append(f"Obstacle at {obs} caused detour.")
            paths.append(new_path)
        else:
            paths.append([-1])
            
    return explanations, paths, obstacles_

# Visualization of single environment with different obstacle/path configurations
def visualize_mosaic(environment, obstacle_sets, paths, grid_size, save_path="mosaic_grid.png"):
    num_variants = len(obstacle_sets)
    cols = 5
    rows = int(np.ceil(num_variants / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    axes = axes.flatten()

    for idx in range(0, num_variants):
        ax = axes[idx]
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        ax.set_title(f"Variant {idx}")
        
        # Plot obstacles
        for obs in obstacle_sets[idx]:
            ax.scatter(obs[1], grid_size - 1 - obs[0], color='black', s=50)

        # Plot agent
        ax.scatter(environment["agent"][1], grid_size - 1 - environment["agent"][0], color='blue', s=50)

        # Plot goal
        ax.scatter(environment["goal"][1], grid_size - 1 - environment["goal"][0], color='green', s=50)

        # Plot replanning path
        path = paths[idx]
        if path != [-1]:
            path_x = [pos[1] for pos in path]
            path_y = [grid_size - 1 - pos[0] for pos in path]
            ax.plot(path_x, path_y, color='orange', linewidth=1.5)

        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def segm_fn(env):
    return

def classifier_fn(env):
    return

def forward_selection(data, labels, weights, num_features):
    """Iteratively adds features to the model"""
    clf = Ridge(alpha=0, fit_intercept=True, random_state=random_state)
    used_features = []
    for _ in range(min(num_features, data.shape[1])):
        max_ = -100000000
        best = 0
        for feature in range(data.shape[1]):
            if feature in used_features:
                continue
            clf.fit(data[:, used_features + [feature]], labels,
                    sample_weight=weights)
            score = clf.score(data[:, used_features + [feature]],
                                labels,
                                sample_weight=weights)
            if score > max_:
                best = feature
                max_ = score
        used_features.append(best)
    return np.array(used_features)

def feature_selection(data, labels, weights, num_features, method):
    """Selects features for the model. see explain_instance_with_data to
        understand the parameters."""
    if method == 'none':
        return np.array(range(data.shape[1]))
    elif method == 'forward_selection':
        return forward_selection(data, labels, weights, num_features)
    elif method == 'highest_weights':
        clf = Ridge(alpha=0.01, fit_intercept=True,
                    random_state=0)
        clf.fit(data, labels, sample_weight=weights)

        coef = clf.coef_
        if sp.sparse.issparse(data):
            coef = sp.sparse.csr_matrix(clf.coef_)
            weighted_data = coef.multiply(data[0])
            # Note: most efficient to slice the data before reversing
            sdata = len(weighted_data.data)
            argsort_data = np.abs(weighted_data.data).argsort()
            # Edge case where data is more sparse than requested number of feature importances
            # In that case, we just pad with zero-valued features
            if sdata < num_features:
                nnz_indexes = argsort_data[::-1]
                indices = weighted_data.indices[nnz_indexes]
                num_to_pad = num_features - sdata
                indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                indices_set = set(indices)
                pad_counter = 0
                for i in range(data.shape[1]):
                    if i not in indices_set:
                        indices[pad_counter + sdata] = i
                        pad_counter += 1
                        if pad_counter >= num_to_pad:
                            break
            else:
                nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                indices = weighted_data.indices[nnz_indexes]
            return indices
        else:
            weighted_data = coef * data[0]
            feature_weights = sorted(
                zip(range(data.shape[1]), weighted_data),
                key=lambda x: np.abs(x[1]),
                reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
    elif method == 'lasso_path':
        weighted_data = ((data - np.average(data, axis=0, weights=weights))
                            * np.sqrt(weights[:, np.newaxis]))
        weighted_labels = ((labels - np.average(labels, weights=weights))
                            * np.sqrt(weights))
        nonzero = range(weighted_data.shape[1])
        _, coefs = generate_lars_path(weighted_data,
                                            weighted_labels)
        for i in range(len(coefs.T) - 1, 0, -1):
            nonzero = coefs.T[i].nonzero()[0]
            if len(nonzero) <= num_features:
                break
        used_features = nonzero
        return used_features
    elif method == 'auto':
        if num_features <= 6:
            n_method = 'forward_selection'
        else:
            n_method = 'highest_weights'
        return feature_selection(data, labels, weights,
                                        num_features, n_method)

def explain_instance(env):
    # define perturbations
    n_features = env.num_obstacles
    
    # all or random perturbations
    test_all_pert = False
    # one obstacle per perturbation
    one_obstacle_per_perturbation = False

    if test_all_pert == True:
        # test all possible combinations
        num_samples = 2 ** n_features
        import itertools
        lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
        data = np.array(lst).reshape((num_samples, n_features))
 
        data[0, :] = 1
        data[-1, :] = 0 # only if I use my perturbation

        #print('data = ', data)
        print('data.shape = ', data.shape)
    else:
        # random perturbations 
        num_samples = 2 ** n_features
        import itertools
        lst = list(map(list, itertools.product([0, 1], repeat=n_features)))
        data_ = np.array(lst).reshape((num_samples, n_features))

        n_pert = 10
        # Select random indices (without replacement)
        random_indices = np.random.choice(len(data_), size=n_pert, replace=False)
        # Extract the selected lists into a new array
        data = data_[random_indices]

        data[0, :] = 1

        #print('data = ', data)
        print('data.shape = ', data.shape)

    if one_obstacle_per_perturbation == True:
        # only 1 obstacle per perturbation
        num_samples = n_features
        lst = [[1]*n_features]
        for i in range(1, num_samples):
            lst.append([1]*n_features)
            lst[i][n_features-i] = 0    
        data = np.array(lst).reshape((num_samples, n_features))
        #print('data = ', data)
        print('data.shape = ', data.shape)
        #to_add = np.array([1]*n_features).reshape(1,n_features)
        #data = np.concatenate((to_add,data))
        #print('data = ', data)    

    # Replanning
    envs = []
    rows = tqdm(data)
    for row in rows:
        temp = copy.deepcopy(env)
        # Filter objects where mask == 1
        obs = copy.deepcopy(temp.obstacles)
        obs_perturbed = [obj for m, obj in zip(row, obs) if m == 1]
        #print('obs_perturbed = ', obs_perturbed)
        temp.obstacles = obs_perturbed
        envs.append(temp)
    
    # Replanning
    paths = []
    grid_size = envs[0].grid_size 
    start = envs[0].agent_pos 
    goal = envs[0].goal_pos
    
    base_path = a_star(grid_size, start, goal, envs[0].obstacles)
    if base_path == None:
        print("\nNo base path found.")
        paths.append([-1])
    else:
        paths.append(base_path)
    
    for env in envs[1:]:
        explanations = []
        new_path = a_star(grid_size, start, goal, env.obstacles)
        if new_path: 
            if base_path == None:
                explanations.append(f"Obstacle at {obs} caused path to be blocked.")
            elif new_path and len(new_path) < len(base_path):
                explanations.append(f"Obstacle at {obs} caused detour.")
            paths.append(new_path)
        else:
            paths.append([-1])

    print('len(paths) = ', len(paths))
    #print('paths = ', paths)

    distance_metric = 'jaccard'
    distances = pairwise_distances(
        data,
        data[0].reshape(1, -1),
        metric=distance_metric
    ).ravel()
    print('distance_metric = ', distance_metric)
    print('distances = ', distances)
    labels = distances

    kernel_width = float(0.25)
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    kernel_fn = partial(kernel, kernel_width=kernel_width)
    weights = kernel_fn(distances)
    random_state = check_random_state(0)
    used_features = feature_selection(data, labels, weights, env.num_obstacles, 'highest_weights')
    model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=random_state)
    easy_model = model_regressor
    easy_model.fit(data[:, used_features], labels, sample_weight=weights)
    prediction_score = easy_model.score(
        data[:, used_features],
        labels, sample_weight=weights)

    local_pred = easy_model.predict(data[0, used_features].reshape(1, -1))

    print('Intercept', easy_model.intercept_)
    print('Prediction_local', local_pred,)
    print('Right:', labels)
    return (easy_model.intercept_,
            sorted(zip(used_features, easy_model.coef_),
                    key=lambda x: np.abs(x[1]), reverse=True),
            prediction_score, local_pred)

# Main experiment loop with GIF generation
if __name__ == "__main__":
    '''
    env = GridWorldEnv(grid_size=10, num_obstacles=40)

    state = env.get_state()
    
    # Replanning
    t0 = time.time()
    replanning_exp, replanning_paths, replanning_obstacles = replanning_explanation(state["agent"], state["goal"], state["obstacles"], state["grid_size"])
    replanning_time = time.time() - t0

    print('replanning_exp: ', replanning_exp)
    print('replanning_time: ', replanning_time)
    #print('replanning_paths: ', replanning_paths)
    
    # Save mosaic grid for the same environment with varying obstacles and paths
    visualize_mosaic(state, replanning_obstacles, replanning_paths, env.grid_size)'
    '''

    env = GridWorldEnv(grid_size=10, num_obstacles=10)

    state = env.get_state()

    # LIME for image
    #explainer = lime_image.LimeImageExplainer(verbose=True)

    #explanation, segments = explainer.explain_instance(env, classifier_fn, batch_size=2048, segmentation_fn=segm_fn, top_labels=10)

    explanation = explain_instance(env)



