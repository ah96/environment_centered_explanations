import heapq
import time
import numpy as np
import math

class ThetaStarPlanner:
    """Theta* algorithm for path planning with any-angle paths"""
    
    def __init__(self, grid_size=10, obstacles=None):
        """Initialize the Theta* planner"""
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
        """Euclidean distance heuristic"""
        return math.sqrt((pos[0] - self.goal[0])**2 + (pos[1] - self.goal[1])**2)
        
    def line_of_sight(self, start, end):
        """Check if there is a clear line of sight between start and end"""
        # Bresenham's line algorithm
        x0, y0 = start[0], start[1]
        x1, y1 = end[0], end[1]
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            # Check if the current cell is an obstacle
            if [x0, y0] in self.obstacles:
                return False
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
            # Check if we're out of bounds
            if not (0 <= x0 < self.grid_size and 0 <= y0 < self.grid_size):
                return False
        
        return True
        
    def plan(self, start=None, goal=None, obstacles=None, return_steps=False):
        """
        Run Theta* algorithm for path planning
        
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
        
        # Initialize variables
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
                "description": "Initializing Theta* algorithm"
            })
        
        while open_set:
            # Get current node
            _, current = heapq.heappop(open_set)
            current_tuple = tuple(current)
            
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
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = [current[0] + dx, current[1] + dy]
                neighbor_tuple = tuple(neighbor)
                
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size and neighbor not in self.obstacles:
                    # Calculate cost - diagonal moves cost more
                    move_cost = 1.0 if dx == 0 or dy == 0 else 1.414
                    
                    if return_steps:
                        neighbor_info = {
                            "pos": neighbor,
                            "g_score": g_score[current_tuple] + move_cost,
                            "h_score": self.h(neighbor),
                            "action": "skip"
                        }
                    
                    # Theta* path smoothing - try to connect to parent's parent if possible
                    if current_tuple in came_from:
                        parent = came_from[current_tuple]
                        parent_tuple = tuple(parent)
                        
                        # Check if there's line of sight from parent to neighbor
                        if self.line_of_sight(parent, neighbor):
                            # Calculate new g_score through parent
                            new_g = g_score[parent_tuple] + math.sqrt(
                                (neighbor[0] - parent[0])**2 + (neighbor[1] - parent[1])**2)
                            
                            if neighbor_tuple not in g_score or new_g < g_score[neighbor_tuple]:
                                g_score[neighbor_tuple] = new_g
                                came_from[neighbor_tuple] = parent
                                f_score = new_g + self.h(neighbor)
                                heapq.heappush(open_set, (f_score, neighbor))
                                
                                if return_steps:
                                    neighbor_info["g_score"] = new_g
                                    neighbor_info["action"] = "add_or_update_line_of_sight"
                            
                            if return_steps:
                                neighbors_data.append(neighbor_info)
                            continue
                    
                    # Standard A* relaxation (if line-of-sight doesn't work)
                    tentative_g = g_score[current_tuple] + move_cost
                    if neighbor_tuple not in g_score or tentative_g < g_score[neighbor_tuple]:
                        came_from[neighbor_tuple] = current
                        g_score[neighbor_tuple] = tentative_g
                        f_score = tentative_g + self.h(neighbor)
                        heapq.heappush(open_set, (f_score, neighbor))
                        
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