import time
import random
import math
from collections import deque

class PRMPlanner:
    def __init__(self, num_samples=200, connection_radius=None):
        self.grid_size = 10
        self.obstacles = []
        self.start = None
        self.goal = None
        self.execution_time = 0
        self.num_samples = num_samples
        self.connection_radius = connection_radius  # Can be set based on grid size

    def set_environment(self, start, goal, grid_size, obstacles):
        self.start = start
        self.goal = goal
        self.grid_size = grid_size
        self.obstacles = set(tuple(o) for o in obstacles)
        if self.connection_radius is None:
            self.connection_radius = max(3.0, grid_size * 0.4)

    def plan(self, return_steps=False):
        start_time = time.time()

        nodes = [self.start, self.goal]
        edges = {}

        steps = []
        if return_steps:
            steps.append({
                "step": 0,
                "type": "init",
                "open_set": [self.start],
                "visited": [],
                "current_path": [],
                "description": "Initializing PRM"
            })

        while len(nodes) < self.num_samples + 2:
            if len(nodes) < 10:
                bias_target = self.start if len(nodes) % 2 == 0 else self.goal
                p = [
                    random.gauss(bias_target[0], 1.5),
                    random.gauss(bias_target[1], 1.5)
                ]
            else:
                p = [random.uniform(0, self.grid_size), random.uniform(0, self.grid_size)]

            if not self._collision(p):
                nodes.append(p)
                if return_steps:
                    steps.append({
                        "step": len(steps),
                        "type": "sample",
                        "current": p,
                        "open_set": list(nodes),
                        "visited": [],
                        "current_path": [],
                        "description": f"Sampled node #{len(nodes)}"
                    })

        for i, node in enumerate(nodes):
            edges[i] = []
            for j, other in enumerate(nodes):
                if i != j and self._distance(node, other) <= self.connection_radius:
                    if not self._collision_line(node, other):
                        edges[i].append(j)
                        if return_steps:
                            steps.append({
                                "step": len(steps),
                                "type": "connect",
                                "current": node,
                                "open_set": list(nodes),
                                "visited": [],
                                "current_path": [],
                                "description": f"Connected node {i} to {j}"
                            })

        # Ensure start and goal are connected if isolated
        start_idx, goal_idx = 0, 1
        for idx, ref_node in [(start_idx, self.start), (goal_idx, self.goal)]:
            if not edges[idx]:
                for i, node in enumerate(nodes[2:], start=2):
                    if self._distance(ref_node, node) <= self.connection_radius and not self._collision_line(ref_node, node):
                        edges[idx].append(i)
                        edges[i].append(idx)

        # BFS search on roadmap
        came_from = {start_idx: None}
        queue = deque([start_idx])

        while queue:
            curr = queue.popleft()
            if curr == goal_idx:
                break
            for neighbor in edges[curr]:
                if neighbor not in came_from:
                    came_from[neighbor] = curr
                    queue.append(neighbor)

                    if return_steps:
                        path_so_far = []
                        idx = neighbor
                        while idx is not None:
                            node = nodes[idx]
                            path_so_far.append([
                                int(round(node[0])),
                                int(round(node[1]))
                            ])
                            idx = came_from.get(idx)
                        path_so_far.reverse()

                        steps.append({
                            "step": len(steps),
                            "type": "explore",
                            "current": nodes[neighbor],
                            "open_set": list(nodes),
                            "visited": [],
                            "current_path": path_so_far,
                            "description": f"Explored node {neighbor}"
                        })

        self.execution_time = time.time() - start_time

        if goal_idx not in came_from:
            return None if not return_steps else (None, steps)

        path = []
        curr = goal_idx
        while curr is not None:
            path.append([
                int(round(nodes[curr][0])),
                int(round(nodes[curr][1]))
            ])
            curr = came_from[curr]
        path.reverse()

        if return_steps:
            steps.append({
                "step": len(steps),
                "type": "success",
                "current": self.goal,
                "open_set": list(nodes),
                "visited": [],
                "current_path": path,
                "description": "Final PRM path"
            })
            return path, steps

        return path

    def _distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _collision(self, p):
        cell = (int(p[0]), int(p[1]))
        return (cell[0] < 0 or cell[0] >= self.grid_size or
                cell[1] < 0 or cell[1] >= self.grid_size or
                cell in self.obstacles)

    def _collision_line(self, a, b):
        steps = max(int(self._distance(a, b) / 0.2), 1)
        for i in range(steps + 1):
            x = a[0] + i / steps * (b[0] - a[0])
            y = a[1] + i / steps * (b[1] - a[1])
            if self._collision([x, y]):
                return True
        return False
