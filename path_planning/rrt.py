import time
import random
import math

class RRTPlanner:
    def __init__(self, max_iterations=500, step_size=1.0, goal_sample_rate=0.05):
        self.grid_size = 10
        self.obstacles = []
        self.start = None
        self.goal = None
        self.execution_time = 0
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate

    def set_environment(self, start, goal, grid_size, obstacles):
        self.start = [float(start[0]), float(start[1])]
        self.goal = [float(goal[0]), float(goal[1])]
        self.grid_size = grid_size
        self.obstacles = set(tuple(o) for o in obstacles)

    def plan(self, return_steps=False):
        start_time = time.time()

        nodes = [self.start]
        parent = {tuple(self._round(self.start)): None}

        steps = []
        if return_steps:
            steps.append({
                "step": 0,
                "type": "init",
                "open_set": [self.start],
                "visited": [],
                "current_path": [],
                "description": "Initializing RRT"
            })

        for i in range(self.max_iterations):
            if random.random() < self.goal_sample_rate:
                rand = self.goal
            else:
                rand = [random.uniform(0, self.grid_size), random.uniform(0, self.grid_size)]

            nearest = min(nodes, key=lambda n: self._distance(n, rand))
            new_node = self._steer(nearest, rand)

            if not self._collision(nearest, new_node):
                nodes.append(new_node)
                parent[tuple(self._round(new_node))] = tuple(self._round(nearest))

                # Check if goal is reached
                if self._distance(new_node, self.goal) < self.step_size * 1.5:
                    goal_node = self.goal
                    parent[tuple(self._round(goal_node))] = tuple(self._round(new_node))
                    path = self._reconstruct_path(parent, goal_node)

                    self.execution_time = time.time() - start_time

                    if return_steps:
                        steps.append({
                            "step": len(steps),
                            "type": "success",
                            "current": new_node,
                            "current_path": path,
                            "description": f"Goal reached in {i} iterations"
                        })
                        return path, steps

                    return path

                # Log step with reconstructed path
                if return_steps:
                    current_path = []
                    curr = tuple(self._round(new_node))
                    while curr is not None:
                        current_path.append(list(curr))
                        curr = parent.get(curr)
                    current_path.reverse()

                    steps.append({
                        "step": len(steps),
                        "type": "explore",
                        "current": new_node,
                        "open_set": list(nodes),
                        "visited": [],
                        "current_path": current_path,
                        "description": f"Added node #{len(nodes)}"
                    })

        self.execution_time = time.time() - start_time
        return None if not return_steps else (None, steps)

    def _round(self, point):
        return [int(round(point[0])), int(round(point[1]))]

    def _distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _steer(self, from_node, to_node):
        angle = math.atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
        return [
            from_node[0] + self.step_size * math.cos(angle),
            from_node[1] + self.step_size * math.sin(angle)
        ]

    def _collision(self, from_node, to_node):
        steps = max(int(self._distance(from_node, to_node) / 0.2), 1)
        for i in range(steps + 1):
            x = from_node[0] + i / steps * (to_node[0] - from_node[0])
            y = from_node[1] + i / steps * (to_node[1] - from_node[1])
            cell = (int(x), int(y))
            if (0 > cell[0] or cell[0] >= self.grid_size or
                0 > cell[1] or cell[1] >= self.grid_size or
                cell in self.obstacles):
                return True
        return False

    def _reconstruct_path(self, parent, goal):
        path = []
        curr = tuple(self._round(goal))
        while curr is not None:
            path.append(list(curr))
            curr = parent.get(curr)
        path.reverse()
        return path
