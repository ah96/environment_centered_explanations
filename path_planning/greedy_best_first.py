import heapq
import time

class GreedyBestFirstPlanner:
    def __init__(self):
        self.grid_size = 10
        self.obstacles = []
        self.start = None
        self.goal = None
        self.execution_time = 0

    def set_environment(self, start, goal, grid_size, obstacles):
        self.start = start
        self.goal = goal
        self.grid_size = grid_size
        self.obstacles = obstacles

    def h(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])  # Manhattan

    def plan(self, return_steps=False):
        start_time = time.time()
        open_set = []
        heapq.heappush(open_set, (self.h(self.start), self.start))
        came_from = {}
        visited = [self.start]

        steps = []
        if return_steps:
            steps.append({
                "step": 0,
                "type": "init",
                "open_set": [self.start],
                "visited": [],
                "current_path": [],
                "description": "Initializing Greedy Best-First Search"
            })

        while open_set:
            _, current = heapq.heappop(open_set)

            step_data = {
                "step": len(steps),
                "type": "explore",
                "current": current,
                "open_set": [n[1] for n in open_set],
                "visited": list(visited),
                "description": f"Exploring {current}"
            }

            # Path so far
            if return_steps:
                if tuple(current) in came_from:
                    path_so_far = []
                    curr = current
                    while tuple(curr) in came_from:
                        path_so_far.append(curr)
                        curr = came_from[tuple(curr)]
                    path_so_far.append(self.start)
                    path_so_far.reverse()
                    step_data["current_path"] = path_so_far
                else:
                    step_data["current_path"] = []

            if current == self.goal:
                path = []
                curr = current
                while tuple(curr) in came_from:
                    path.append(curr)
                    curr = came_from[tuple(curr)]
                path.append(self.start)
                path.reverse()

                if return_steps:
                    step_data["type"] = "success"
                    step_data["current_path"] = path
                    step_data["description"] = f"Goal reached! Path length: {len(path)}"
                    steps.append(step_data)

                self.execution_time = time.time() - start_time
                return path if not return_steps else (path, steps)

            visited.append(current)

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = [current[0] + dx, current[1] + dy]
                if (0 <= neighbor[0] < self.grid_size and
                    0 <= neighbor[1] < self.grid_size and
                    neighbor not in self.obstacles and
                    neighbor not in visited):

                    heapq.heappush(open_set, (self.h(neighbor), neighbor))
                    came_from[tuple(neighbor)] = current
                    visited.append(neighbor)

            if return_steps:
                steps.append(step_data)

        self.execution_time = time.time() - start_time
        return None if not return_steps else (None, steps)
