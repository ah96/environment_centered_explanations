import time
import random
import math

class RRTStarPlanner:
    def __init__(self, max_iterations=500, step_size=1.0, goal_sample_rate=0.05, radius=2.0):
        self.grid_size = 10
        self.obstacles = []
        self.start = None
        self.goal = None
        self.execution_time = 0
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.radius = radius

    def set_environment(self, start, goal, grid_size, obstacles):
        self.start = [float(start[0]), float(start[1])]
        self.goal = [float(goal[0]), float(goal[1])]
        self.grid_size = grid_size
        self.obstacles = set(tuple(o) for o in obstacles)

    def plan(self, start=None, goal=None, obstacles=None, return_steps=False):
        """
        Run RRT* path planning algorithm
        
        Args:
            start: Start position [row, col], uses stored start if None
            goal: Goal position [row, col], uses stored goal if None
            obstacles: List of obstacle positions, uses stored obstacles if None
            return_steps: If True, returns planning steps for visualization
            
        Returns:
            If return_steps is False: path or None (if no path found)
            If return_steps is True: (path, steps) or (None, steps)
        """
        # Update parameters if provided
        if start is not None:
            self.start = [float(start[0]), float(start[1])]
        if goal is not None:
            self.goal = [float(goal[0]), float(goal[1])]
        if obstacles is not None:
            self.obstacles = set(tuple(o) for o in obstacles)
            
        # Verify we have valid start and goal
        if not self.start or not self.goal:
            return None if not return_steps else (None, [])
            
        start_time = time.time()

        nodes = [self.start]
        parent = {tuple(self._round(self.start)): None}
        cost = {tuple(self._round(self.start)): 0}

        steps = []
        if return_steps:
            steps.append({
                "step": 0,
                "type": "init",
                "open_set": [self.start],
                "visited": [],
                "current_path": [],
                "description": "Initializing RRT*"
            })

        for i in range(self.max_iterations):
            rand = self.goal if random.random() < self.goal_sample_rate else [
                random.uniform(0, self.grid_size),
                random.uniform(0, self.grid_size)
            ]

            nearest = min(nodes, key=lambda n: self._distance(n, rand))
            new_node = self._steer(nearest, rand)

            if self._collision(nearest, new_node):
                continue

            nodes.append(new_node)
            new_node_r = tuple(self._round(new_node))
            nearest_r = tuple(self._round(nearest))

            # Default connection
            parent[new_node_r] = nearest_r
            cost[new_node_r] = cost[nearest_r] + self._distance(nearest, new_node)

            # Rewire: try to find better parent
            for node in nodes:
                node_r = tuple(self._round(node))
                if node_r == new_node_r:
                    continue
                if self._distance(node, new_node) <= self.radius and not self._collision(node, new_node):
                    new_cost = cost[node_r] + self._distance(node, new_node)
                    if new_cost < cost[new_node_r]:
                        parent[new_node_r] = node_r
                        cost[new_node_r] = new_cost

            # Check goal connection
            if self._distance(new_node, self.goal) < self.step_size * 1.5:
                goal_r = tuple(self._round(self.goal))
                parent[goal_r] = new_node_r
                cost[goal_r] = cost[new_node_r] + self._distance(new_node, self.goal)
                path = self._reconstruct_path(parent, self.goal)

                self.execution_time = time.time() - start_time

                if return_steps:
                    steps.append({
                        "step": len(steps),
                        "type": "success",
                        "current": self.goal,
                        "current_path": path,
                        "open_set": list(nodes),
                        "description": f"Goal reached in {i} iterations"
                    })
                    return path, steps

                return path

            # Step recording with path reconstruction
            if return_steps:
                path_so_far = []
                curr = new_node_r
                while curr is not None:
                    path_so_far.append(list(curr))
                    curr = parent.get(curr)
                path_so_far.reverse()

                steps.append({
                    "step": len(steps),
                    "type": "explore",
                    "current": new_node,
                    "current_path": path_so_far,
                    "open_set": list(nodes),
                    "visited": [],
                    "description": f"Added node #{len(nodes)} (rewired if needed)"
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
            if (cell[0] < 0 or cell[0] >= self.grid_size or
                cell[1] < 0 or cell[1] >= self.grid_size or
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
