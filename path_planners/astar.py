import heapq
import time
import numpy as np

class AStarPlanner:
    """A* path planning algorithm implementation"""
    
    def __init__(self, grid_size=10, obstacles=None):
        """Initialize the A* planner with environment parameters"""
        self.grid_size = grid_size
        self.obstacles = obstacles or []
        self.start = None
        self.goal = None
        self.execution_time = 0
        
    def set_environment(self, start, goal, grid_size, obstacles):
        """Set or update the environment for planning"""
        self.start = start
        self.goal = goal
        self.grid_size = grid_size
        self.obstacles = obstacles
        
    def h(self, pos):
        """Heuristic function - Manhattan distance"""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
    
    def plan(self, start=None, goal=None, obstacles=None, return_steps=False):
        """
        Run A* path planning algorithm
        
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
            self.start = start
        if goal is not None:
            self.goal = goal
        if obstacles is not None:
            self.obstacles = obstacles
            
        # Verify we have valid start and goal
        if not self.start or not self.goal:
            return None if not return_steps else (None, [])
            
        # Start timer
        start_time = time.time()
        
        # Initialize A* variables
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {tuple(self.start): 0}
        
        # For visualization
        visited = []
        current_path = []
        
        # For step tracking if needed
        steps = []
        if return_steps:
            # Record initial state
            steps.append({
                "step": 0,
                "type": "init",
                "open_set": [self.start],
                "current": None,
                "visited": [],
                "current_path": [],
                "description": "Initializing A* algorithm"
            })
        
        while open_set:
            # Get current node
            _, current = heapq.heappop(open_set)
            
            # Record step data if tracking steps
            if return_steps:
                step_data = {
                    "step": len(steps),
                    "type": "explore",
                    "current": current,
                    "open_set": [list(n[1]) for n in open_set],
                    "visited": list(visited),
                    "g_score": {str(k): v for k, v in g_score.items()},
                    "description": f"Exploring node at {current}"
                }
            
            # Check if goal reached
            if current == self.goal:
                # Reconstruct path
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
            
            # Add current to visited
            visited.append(current)
            
            # Record the current path for visualization if tracking steps
            if return_steps and tuple(current) in came_from:
                curr = current
                current_path = []
                while tuple(curr) in came_from:
                    current_path.append(curr)
                    curr = came_from[tuple(curr)]
                current_path.append(self.start)
                current_path.reverse()
                step_data["current_path"] = current_path
            
            # Explore neighbors
            neighbors_data = [] if return_steps else None
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = [current[0] + dx, current[1] + dy]
                
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size and neighbor not in self.obstacles:
                    tentative_g = g_score[tuple(current)] + 1
                    
                    if return_steps:
                        neighbor_info = {
                            "pos": neighbor,
                            "g_score": tentative_g,
                            "h_score": self.h(neighbor),
                            "f_score": tentative_g + self.h(neighbor),
                            "action": "skip"
                        }
                    
                    if tuple(neighbor) not in g_score or tentative_g < g_score[tuple(neighbor)]:
                        came_from[tuple(neighbor)] = current
                        g_score[tuple(neighbor)] = tentative_g
                        heapq.heappush(open_set, (tentative_g + self.h(neighbor), neighbor))
                        
                        if return_steps:
                            neighbor_info["action"] = "add_or_update"
                    
                    if return_steps:
                        neighbors_data.append(neighbor_info)
            
            if return_steps:
                step_data["neighbors"] = neighbors_data
                steps.append(step_data)
        
        # If we get here, no path was found
        self.execution_time = time.time() - start_time
        return None if not return_steps else (None, steps)