import numpy as np
import heapq
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from datetime import datetime
import os
import random
from sklearn.linear_model import Ridge
from matplotlib import colors

# Enhanced GridWorld environment with shaped obstacles
class GridWorldEnv:
    def __init__(self, grid_size=10, num_obstacles=5):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.agent_pos = None  # Will be set by user
        self.goal_pos = None   # Will be set by user
        # Dictionary to store obstacle shapes: {shape_id: list_of_positions}
        self.obstacle_shapes = {}  
        # List of all obstacle positions for quick lookup
        self.obstacles = []    
        self.generate_obstacles()
        return self.get_state()

    def generate_obstacles(self):
        self.obstacle_shapes = {}
        self.obstacles = []
        
        # Divide grid into regions for obstacles
        regions = self.divide_grid_into_regions()
        
        # Generate an obstacle shape in each region
        for shape_id, region in enumerate(regions):
            if shape_id >= self.num_obstacles:
                break
                
            # Generate a connected shape within this region
            shape_points = self.generate_connected_shape(region)
            
            # Store the shape with its ID
            self.obstacle_shapes[shape_id] = shape_points
            
            # Add all points to obstacles list for quick lookup
            self.obstacles.extend(shape_points)
    
    def divide_grid_into_regions(self):
        """Divide the grid into regions for placing obstacles"""
        regions = []
        
        # Simple approach: divide into approximately equal quadrants
        region_size = max(3, self.grid_size // 3)
        
        for row_start in range(0, self.grid_size, region_size):
            for col_start in range(0, self.grid_size, region_size):
                row_end = min(row_start + region_size, self.grid_size)
                col_end = min(col_start + region_size, self.grid_size)
                
                region = {
                    'min_row': row_start,
                    'max_row': row_end - 1,
                    'min_col': col_start,
                    'max_col': col_end - 1
                }
                regions.append(region)
        
        # Shuffle regions to get random distribution
        random.shuffle(regions)
        return regions
    
    def generate_connected_shape(self, region):
        """Generate a randomly connected shape within a region"""
        min_row, max_row = region['min_row'], region['max_row']
        min_col, max_col = region['min_col'], region['max_col']
        
        # Determine shape size (number of cells)
        # Ensure it's reasonable for the region size
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        max_size = max(1, min(width * height // 3, 8))  # Limit max size
        shape_size = random.randint(1, max_size)
        
        # Start with a random point in the region
        start_row = random.randint(min_row, max_row)
        start_col = random.randint(min_col, max_col)
        shape_points = [[start_row, start_col]]
        
        # Add connected points
        attempts = 0
        while len(shape_points) < shape_size and attempts < 50:
            # Get last point added
            last_point = shape_points[-1]
            
            # Try to add a neighbor
            neighbors = [
                [last_point[0]-1, last_point[1]],  # up
                [last_point[0]+1, last_point[1]],  # down
                [last_point[0], last_point[1]-1],  # left
                [last_point[0], last_point[1]+1]   # right
            ]
            
            # Filter valid neighbors (within region and not already in shape)
            valid_neighbors = [
                n for n in neighbors 
                if min_row <= n[0] <= max_row and 
                   min_col <= n[1] <= max_col and 
                   n not in shape_points
            ]
            
            if valid_neighbors:
                # Add a random valid neighbor
                new_point = random.choice(valid_neighbors)
                shape_points.append(new_point)
                attempts = 0
            else:
                # No valid neighbors, backtrack and try from another point
                attempts += 1
                if shape_points:
                    # Choose a random existing point as new last point
                    shape_points[-1] = random.choice(shape_points)
        
        return shape_points
    
    def remove_obstacle_shape(self, shape_id):
        """Remove an obstacle shape by its ID"""
        if shape_id in self.obstacle_shapes:
            # Remove all points of this shape from obstacles list
            points_to_remove = self.obstacle_shapes[shape_id]
            # Use list comprehension for potentially better performance on large lists
            self.obstacles = [p for p in self.obstacles if p not in points_to_remove]
            # Keep the shape definition in self.obstacle_shapes for restoration
            return True
        return False
    
    def generate_perturbation_combinations(self, strategy="random"):
        """
        Generates combinations of obstacle removals based on specified strategy.
        
        Args:
            strategy (str): The combination generation strategy. Options:
                "full_combinations": All possible binary combinations of obstacles.
                "random": One random combination.
                "remove_each_obstacle_once": N combinations, each with one obstacle removed.
                
        Returns:
            list: List of combinations, where each combination is a list of 0s and 1s
                (0 means obstacle removed, 1 means obstacle kept).
        """
        all_shape_ids = list(self.obstacle_shapes.keys())
        n = len(all_shape_ids)
        
        if n == 0:  # No obstacles
            return [[]]
            
        if strategy == "full_combinations":
            # Generate all 2^n combinations using binary representation
            combinations = []
            for i in range(2**n):
                # Convert number to binary and pad with leading zeros
                binary = format(i, f'0{n}b')
                # Convert to list of integers (0s and 1s)
                combination = [int(bit) for bit in binary]
                combinations.append(combination)
            return combinations
            
        elif strategy == "random":
            # Return one random combination
            import random
            combination = [random.randint(0, 1) for _ in range(n)]
            return [combination]  # Return as a list of combinations
            
        elif strategy == "remove_each_obstacle_once":
            # Generate n combinations, each removing exactly one obstacle
            combinations = []
            for i in range(n):
                combination = [1] * n  # Start with all obstacles kept
                combination[i] = 0     # Remove just one obstacle
                combinations.append(combination)
            return combinations
        
        else:
            raise ValueError(f"Unknown combination strategy: {strategy}")

    def generate_perturbation(self, strategy="random", combination=None):
        """
        Generates and applies perturbation by removing obstacle shapes based on strategy and combination.
        
        Args:
            strategy (str): The perturbation strategy.
            combination (list, optional): A specific combination to apply (list of 0s and 1s).
                If provided, this overrides the strategy.
                
        Returns:
            tuple: (original_obstacles, shapes_removed_ids)
                original_obstacles (list): A copy of the obstacles list before perturbation.
                shapes_removed_ids (list): A list of the IDs of the obstacle shapes that were removed.
        """
        # Create a copy of the current obstacles for reverting later
        original_obstacles = self.obstacles.copy()
        all_shape_ids = list(self.obstacle_shapes.keys())
        shapes_to_remove_ids = []
        
        if not all_shape_ids:  # No obstacles to remove
            return original_obstacles, []
        
        # If a specific combination is provided, use it
        if combination is not None:
            # Ensure combination is the correct length
            if len(combination) != len(all_shape_ids):
                raise ValueError(f"Combination length {len(combination)} doesn't match obstacle count {len(all_shape_ids)}")
            
            # Remove obstacles where combination has 0
            for i, keep in enumerate(combination):
                if keep == 0:
                    shapes_to_remove_ids.append(all_shape_ids[i])
                    
        # Otherwise use the strategy from before    
        elif strategy == "randomly":
            # Randomly select shapes to remove (approximately 30% of shapes)
            # Ensure at least one is potentially removed if obstacles exist
            num_to_remove = max(1, len(all_shape_ids) // 3)
            # Ensure sample size doesn't exceed population size
            k = min(num_to_remove, len(all_shape_ids))
            if k > 0:  # Only sample if k is positive
                shapes_to_remove_ids = random.sample(all_shape_ids, k)
                
        elif strategy == "each_obstacle_once":
            # Remove exactly one randomly chosen obstacle shape
            if all_shape_ids:
                shape_to_remove_id = random.choice(all_shape_ids)
                shapes_to_remove_ids = [shape_to_remove_id]
                
        elif strategy == "full_perturbation":
            # Remove all obstacles
            shapes_to_remove_ids = list(all_shape_ids)
            
        else:
            raise ValueError(f"Unknown perturbation strategy: {strategy}")
        
        # Apply the perturbation by removing the selected obstacle shapes
        for shape_id in shapes_to_remove_ids:
            self.remove_obstacle_shape(shape_id)
            
        return original_obstacles, shapes_to_remove_ids

    def restore_from_perturbation(self, original_obstacles):
        """Restore the environment to the state before perturbation"""
        # Simply reset the obstacles list to the saved original state.
        self.obstacles = original_obstacles.copy()


    def get_state(self):
        return {
            "grid_size": self.grid_size,
            "agent": self.agent_pos,
            "goal": self.goal_pos,
            "obstacles": self.obstacles,
            "obstacle_shapes": self.obstacle_shapes
        }

# Path Planning App
class PathPlanningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Path Planning Simulator")
        self.root.geometry("900x800")
        
        # Environment settings
        self.grid_size = 10
        self.num_obstacles = 8
        self.env = None
        self.algorithm_steps = []
        self.current_step = 0
        self.animation_speed = 200  # milliseconds
        self.animation_running = False
        
        # Interaction settings
        self.selected_obstacle = None
        self.selected_shape_id = None
        
        # Algorithm options
        self.algorithms = ["A*"]  # Will add more algorithms later
        self.selected_algorithm = tk.StringVar(value=self.algorithms[0])
        
        # Explainability options
        self.explainability_methods = ["None", "Basic", "Advanced"]
        self.selected_explainability = tk.StringVar(value=self.explainability_methods[0])
        
        # Create GUI components
        self.create_gui()
        
        # Generate initial environment
        self.generate_environment()

    def create_gui(self):
        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Add settings panel for grid size and obstacles
        settings_frame = ttk.LabelFrame(self.root, text="Environment Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=10, pady=5, before=control_frame)
        
        # Grid size control
        ttk.Label(settings_frame, text="Grid Size:").grid(row=0, column=0, padx=5, pady=5)
        self.grid_size_var = tk.IntVar(value=self.grid_size)
        grid_size_spin = ttk.Spinbox(settings_frame, from_=5, to=20, textvariable=self.grid_size_var, width=5)
        grid_size_spin.grid(row=0, column=1, padx=5, pady=5)
        
        # Obstacles count control
        ttk.Label(settings_frame, text="Obstacles:").grid(row=0, column=2, padx=5, pady=5)
        self.num_obstacles_var = tk.IntVar(value=self.num_obstacles)
        obstacles_spin = ttk.Spinbox(settings_frame, from_=0, to=30, textvariable=self.num_obstacles_var, width=5)
        obstacles_spin.grid(row=0, column=3, padx=5, pady=5)
        
        # Apply button
        ttk.Button(settings_frame, text="Apply Settings", 
                command=self.apply_settings).grid(row=0, column=4, padx=5, pady=5)
        
        # First row
        ttk.Button(control_frame, text="Generate Environment", 
                command=self.generate_environment).grid(row=0, column=0, padx=5, pady=5)  
        
        ttk.Label(control_frame, text="Algorithm:").grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Combobox(control_frame, textvariable=self.selected_algorithm, 
                     values=self.algorithms, state="readonly").grid(row=0, column=2, padx=5, pady=5)
        
        # Add explainability method selection
        ttk.Label(control_frame, text="Explainability:").grid(row=0, column=3, padx=5, pady=5)
        ttk.Combobox(control_frame, textvariable=self.selected_explainability, 
                     values=self.explainability_methods, state="readonly").grid(row=0, column=4, padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Planning", 
                                       command=self.start_planning, state=tk.DISABLED)
        self.start_button.grid(row=0, column=5, padx=5, pady=5)

        explain_button = tk.Button(control_frame, text="Explain", command=self.explain)
        explain_button.grid(row=7, column=0, pady=5)
        
        # Second row - Status and info
        ttk.Label(control_frame, text="Status:").grid(row=1, column=0, pady=5, sticky="w")
        
        self.status_var = tk.StringVar(value="Click on empty space to set start position")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=1, columnspan=7, pady=5, sticky="w")
        
        # Execution time display
        ttk.Label(control_frame, text="Execution time:").grid(row=2, column=0, pady=5, sticky="w")
        self.exec_time_var = tk.StringVar(value="N/A")
        ttk.Label(control_frame, textvariable=self.exec_time_var).grid(row=2, column=1, columnspan=7, pady=5, sticky="w")
        
        # Canvas for grid display
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect mouse click and drag events
        self.canvas_widget.mpl_connect('button_press_event', self.on_grid_click)
        self.canvas_widget.mpl_connect('motion_notify_event', self.on_grid_drag)
        self.canvas_widget.mpl_connect('button_release_event', self.on_grid_release)
    
    def apply_settings(self):
        # Update internal settings
        self.grid_size = self.grid_size_var.get()
        self.num_obstacles = self.num_obstacles_var.get()
        
        # Regenerate environment with new settings
        self.generate_environment()
        self.status_var.set(f"Settings applied: Grid size {self.grid_size}, Obstacles {self.num_obstacles}")

    def on_grid_click(self, event):
        if self.animation_running or event.xdata is None or event.ydata is None:
            return
            
        # Convert click coordinates to grid position
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        
        # Check if position is within grid
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            position = [row, col]
            
            # Check if the click is on an obstacle
            is_obstacle = False
            for shape_id, points in self.env.obstacle_shapes.items():
                if position in points:
                    is_obstacle = True
                    # Handle as obstacle click - select for moving
                    self.selected_obstacle = position
                    self.selected_shape_id = shape_id
                    self.status_var.set(f"Selected obstacle at {position}. Drag to move it.")
                    self.draw_grid()  # Redraw to show selection
                    break
            
            # If not an obstacle, handle as empty space click - set points
            if not is_obstacle:                
                # Set start position
                if self.env.agent_pos is None:
                    self.env.agent_pos = position
                    self.status_var.set("Click on empty space to set goal position")
                    self.draw_grid()
                    return
                        
                # Set goal position
                if self.env.goal_pos is None and position != self.env.agent_pos:
                    self.env.goal_pos = position
                    self.status_var.set("Ready to start planning")
                    self.start_button.config(state=tk.NORMAL)
                    self.draw_grid()
                    return

    def on_grid_drag(self, event):
        # Early return checks
        if (self.animation_running or 
            self.selected_obstacle is None or 
            self.selected_shape_id is None or
            event.xdata is None or 
            event.ydata is None):
            return
            
        # Convert coordinates to grid position
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        
        # Calculate movement delta from original selected obstacle position
        old_position = self.selected_obstacle
        delta_row = row - old_position[0]
        delta_col = col - old_position[1]
        
        # Get all points in the current shape
        shape_points = self.env.obstacle_shapes[self.selected_shape_id].copy()
        
        # Calculate new positions for all points in the shape
        new_positions = []
        for point in shape_points:
            new_row = point[0] + delta_row
            new_col = point[1] + delta_col
            new_positions.append([new_row, new_col])
        
        # Check if all new positions are valid
        valid_move = True
        for new_pos in new_positions:
            # Check if position is within grid boundaries
            if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                valid_move = False
                break
                
            # Check if not overlapping with other obstacle shapes
            for other_shape_id, other_points in self.env.obstacle_shapes.items():
                if other_shape_id != self.selected_shape_id and new_pos in other_points:
                    valid_move = False
                    break
                    
            # Check if not overlapping with start/goal positions
            if (self.env.agent_pos and new_pos == self.env.agent_pos) or \
            (self.env.goal_pos and new_pos == self.env.goal_pos):
                valid_move = False
                break
            
            # If any check failed, stop validating further positions
            if not valid_move:
                break
        
        # Update positions only if the move is valid
        if valid_move:
            # First, remove all current shape points from obstacles list
            for point in shape_points:
                if point in self.env.obstacles:
                    self.env.obstacles.remove(point)
            
            # Update the shape with new positions
            self.env.obstacle_shapes[self.selected_shape_id] = new_positions
            
            # Add all new positions to obstacles list
            self.env.obstacles.extend(new_positions)
            
            # Update selected obstacle to maintain the same relative position in the shape
            idx = shape_points.index(self.selected_obstacle)
            self.selected_obstacle = new_positions[idx]
            
            # Redraw the grid
            self.draw_grid()
            
            # Set status message
            self.status_var.set(f"Moving obstacle shape #{self.selected_shape_id}")
        else:
            # Inform user that the move is not valid
            self.status_var.set("Cannot move here - out of bounds or obstacle collision")

    def on_grid_release(self, event):
        if self.selected_obstacle is not None:
            # Reset selection after moving
            self.selected_obstacle = None
            self.selected_shape_id = None
            
            # Update status - don't recalculate path automatically
            if self.env.agent_pos is not None and self.env.goal_pos is not None:
                # Enable the start button so user can recalculate when ready
                self.start_button.config(state=tk.NORMAL)
                self.status_var.set("Obstacle moved. Click 'Start Planning' to recalculate path.")
            else:
                if self.env.agent_pos is None:
                    self.status_var.set("Obstacle moved. Click on empty space to set start position.")
                else:
                    self.status_var.set("Obstacle moved. Click on empty space to set goal position.")

    def generate_environment(self):
        self.env = GridWorldEnv(grid_size=self.grid_size, num_obstacles=self.num_obstacles)
        self.env.agent_pos = None
        self.env.goal_pos = None
        self.algorithm_steps = []
        self.current_step = 0
        self.animation_running = False
        self.status_var.set("Click on empty space to set start position")
        self.exec_time_var.set("N/A")
        self.start_button.config(state=tk.DISABLED)
        self.draw_grid()

    def draw_grid(self):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(np.arange(0, self.grid_size, 1))
        self.ax.set_yticks(np.arange(0, self.grid_size, 1))
        self.ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_title("Path Planning Environment")
        
        # Draw shaped obstacles - use different colors for different shapes
        state = self.env.get_state()
        
        # Generate a color map for obstacle shapes
        cmap = plt.colormaps['tab10']
        
        for shape_id, points in state["obstacle_shapes"].items():
            color = cmap(shape_id % 10)  # Use modulo to avoid index errors with many shapes
            
            # Draw filled shape background first
            if points:
                # Add background patches to show connected shapes
                for point in points:
                    # Create a rectangular patch for each cell
                    rect = plt.Rectangle((point[1]-0.5, point[0]-0.5), 1, 1, 
                                        color=color, alpha=0.3)
                    self.ax.add_patch(rect)
            
            # Then draw the obstacle points on top
            for point in points:
                # Highlight selected obstacle
                if self.selected_obstacle == point and self.selected_shape_id == shape_id:
                    self.ax.scatter(point[1], point[0], color='purple', s=120, marker='s')
                else:
                    self.ax.scatter(point[1], point[0], color=color, s=100, marker='s')
            
            # Add a label for the first point of each shape
            if points:
                first_point = points[0]
                self.ax.annotate(f"#{shape_id}", 
                                (first_point[1], first_point[0]),
                                color='white', fontsize=8, 
                                ha='center', va='center')

        # Draw agent if set
        if self.env.agent_pos:
            self.ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0], 
                            color='blue', s=150, marker='o', label="Start")
        
        # Draw goal if set
        if self.env.goal_pos:
            self.ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0], 
                            color='green', s=150, marker='*', label="Goal")
        
        if self.env.agent_pos or self.env.goal_pos:
            self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        # Invert y-axis to make (0,0) at top-left
        self.ax.invert_yaxis()
            
        self.canvas_widget.draw()

    def start_planning(self):
        if self.env.agent_pos is None or self.env.goal_pos is None:
            self.status_var.set("Please set start and goal positions")
            return
            
        algorithm = self.selected_algorithm.get()
        self.status_var.set(f"Running {algorithm}...")
        self.exec_time_var.set("Running...")
        self.root.update()  # Force update the GUI to show status
        
        self.start_button.config(state=tk.DISABLED)
        
        # Start the algorithm and record start time
        #self.start_time = time.time() # now implemented inside a run_astar_visualization function
        
        if algorithm == "A*":
            self.run_astar_visualization()
        
        # Note: We'll update execution time when the algorithm actually completes
        # in the process_step function when animation_running becomes False

    def run_astar_visualization(self):
        self.start_time = time.time()

        # Prepare for recording steps
        self.algorithm_steps = []
        state = self.env.get_state()
        start = state["agent"]
        goal = state["goal"] 
        obstacles = state["obstacles"]
        grid_size = state["grid_size"]
        
        # Initialize A* variables
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {tuple(start): 0}
        
        # For visualization
        visited = []
        current_path = []
        
        def h(pos):
            # Manhattan distance heuristic
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Record initial state
        self.algorithm_steps.append({
            "step": len(self.algorithm_steps),
            "type": "init",
            "open_set": [start],
            "current": None,
            "visited": [],
            "current_path": [],
            "description": "Initializing A* algorithm"
        })
        
        # Start algorithm step by step
        self.animation_running = True
        self.current_step = 0
        
        def process_step():
            nonlocal open_set, visited, current_path
            
            # Check if we should stop the algorithm
            if not open_set:
                # No path found case
                self.animation_running = False
                self.status_var.set("No path found!")
                
                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - self.start_time
                self.exec_time_var.set(f"{execution_time:.4f} seconds")
                
                # Save steps
                self.save_algorithm_steps(execution_time)
                
                # Can't generate mosaic without path
                self.root.update()
                return
            
            # Get current node
            _, current = heapq.heappop(open_set)
            
            # Record step data
            step_data = {
                "step": len(self.algorithm_steps),
                "type": "explore",
                "current": current,
                "open_set": [list(n[1]) for n in open_set],
                "visited": list(visited),
                "g_score": {str(k): v for k, v in g_score.items()},
                "description": f"Exploring node at {current}"
            }
            
            # Check if goal reached
            if current == goal:
                # Reconstruct path
                current_path = []
                curr = current
                while tuple(curr) in came_from:
                    current_path.append(curr)
                    curr = came_from[tuple(curr)]
                current_path.append(start)
                current_path.reverse()
                
                step_data["type"] = "success"
                step_data["current_path"] = current_path
                step_data["description"] = f"Goal reached! Path length: {len(current_path)}"
                self.algorithm_steps.append(step_data)
                
                # Draw final path
                self.draw_step(len(self.algorithm_steps) - 1)
                
                # Important: Stop the animation and perform finishing tasks
                self.animation_running = False
                self.status_var.set(f"Path found! Length: {len(current_path)}")
                
                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - self.start_time
                self.exec_time_var.set(f"{execution_time:.4f} seconds")
                
                # Save steps and generate visualization
                self.save_algorithm_steps(execution_time)
                self.generate_mosaic_visualization()
                
                # Make sure the GUI updates
                self.root.update()
                return
                
            # Add current to visited
            visited.append(current)
            
            # Record the current path for visualization
            if tuple(current) in came_from:
                curr = current
                current_path = []
                while tuple(curr) in came_from:
                    current_path.append(curr)
                    curr = came_from[tuple(curr)]
                current_path.append(start)
                current_path.reverse()
                step_data["current_path"] = current_path
            
            # Explore neighbors
            neighbors_data = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = [current[0] + dx, current[1] + dy]
                
                if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and neighbor not in obstacles:
                    tentative_g = g_score[tuple(current)] + 1
                    
                    neighbor_info = {
                        "pos": neighbor,
                        "g_score": tentative_g,
                        "h_score": h(neighbor),
                        "f_score": tentative_g + h(neighbor),
                        "action": "skip"
                    }
                    
                    if tuple(neighbor) not in g_score or tentative_g < g_score[tuple(neighbor)]:
                        came_from[tuple(neighbor)] = current
                        g_score[tuple(neighbor)] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
                        neighbor_info["action"] = "add_or_update"
                        
                    neighbors_data.append(neighbor_info)
            
            step_data["neighbors"] = neighbors_data
            self.algorithm_steps.append(step_data)
            
            # Update visualization
            self.draw_step(len(self.algorithm_steps) - 1)
            
            # Schedule next step only if animation still running
            if self.animation_running:
                self.root.after(self.animation_speed, process_step)
        
        # Start the processing loop
        self.root.after(100, process_step)
    
    def run_astar_for_analysis(self):
        """Non-visual A* implementation for analysis purposes"""
        state = self.env.get_state()
        start = state["agent"]
        goal = state["goal"] 
        obstacles = state["obstacles"]
        grid_size = state["grid_size"]
        
        if not start or not goal:
            return None  # No path possible if start or goal not set
        
        # Manhattan distance heuristic
        def h(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Initialize A* variables
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {tuple(start): 0}
        
        while open_set:
            # Get current node
            _, current = heapq.heappop(open_set)
            
            # Check if goal reached
            if current == goal:
                # Reconstruct path
                path = []
                curr = current
                while tuple(curr) in came_from:
                    path.append(curr)
                    curr = came_from[tuple(curr)]
                path.append(start)
                path.reverse()
                return path  # Return the path directly
            
            # Explore neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = [current[0] + dx, current[1] + dy]
                
                if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and neighbor not in obstacles:
                    tentative_g = g_score[tuple(current)] + 1
                    
                    if tuple(neighbor) not in g_score or tentative_g < g_score[tuple(neighbor)]:
                        came_from[tuple(neighbor)] = current
                        g_score[tuple(neighbor)] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))
        
        return None  # No path found

    def draw_step(self, step_idx):
        if step_idx >= len(self.algorithm_steps):
            return
            
        step_data = self.algorithm_steps[step_idx]
        self.current_step = step_idx
        
        # Draw grid
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(np.arange(0, self.grid_size, 1))
        self.ax.set_yticks(np.arange(0, self.grid_size, 1))
        self.ax.grid(True)
        
        state = self.env.get_state()
        
        # Draw shaped obstacles with different colors
        cmap = plt.colormaps['tab10']
        for shape_id, points in state["obstacle_shapes"].items():
            color = cmap(shape_id % 10)  # Added modulo to ensure consistency
            
            # Draw filled shape background first
            if points:
                # Add background patches to show connected shapes
                for point in points:
                    # Create a rectangular patch for each cell
                    rect = plt.Rectangle((point[1]-0.5, point[0]-0.5), 1, 1, 
                                        color=color, alpha=0.3)
                    self.ax.add_patch(rect)
            
            # Then draw the obstacle points
            for point in points:
                self.ax.scatter(point[1], point[0], color=color, s=80, marker='s')
        
        # Draw visited nodes
        if "visited" in step_data and step_data["visited"]:
            for node in step_data["visited"]:
                self.ax.scatter(node[1], node[0], color='gray', alpha=0.5, s=80)
        
        # Draw open set
        if "open_set" in step_data and step_data["open_set"]:
            for node in step_data["open_set"]:
                self.ax.scatter(node[1], node[0], color='orange', alpha=0.7, s=80)
        
        # Draw current node
        if "current" in step_data and step_data["current"]:
            current = step_data["current"]
            self.ax.scatter(current[1], current[0], color='red', s=100)
        
        # Draw current path
        if "current_path" in step_data and step_data["current_path"]:
            path = step_data["current_path"]
            path_x = [node[1] for node in path]
            path_y = [node[0] for node in path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=2)
        
        # Draw start and goal
        self.ax.scatter(state["agent"][1], state["agent"][0], 
                        color='blue', s=150, marker='o', label="Start")
        self.ax.scatter(state["goal"][1], state["goal"][0], 
                        color='green', s=150, marker='*', label="Goal")
        
        # Add title with step info
        self.ax.set_title(f"Step {step_idx}: {step_data['description']}")
        
        # Add legend
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Invert y-axis to make (0,0) at top-left
        self.ax.invert_yaxis()
        
        # Update canvas
        self.canvas_widget.draw()

    def save_algorithm_steps(self, execution_time):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Fix filename - replace asterisk with text
        algorithm_name = self.selected_algorithm.get().replace("*", "Star")
        
        # Create output directories if they don't exist
        json_dir = "output_json"
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        
        filename = os.path.join(json_dir, f"path_planning_{algorithm_name}_{timestamp}.json")
        
        # Add execution time and final results
        summary = {
            "algorithm": self.selected_algorithm.get(),
            "explainability": self.selected_explainability.get(),
            "grid_size": self.grid_size,
            "start": self.env.agent_pos,
            "goal": self.env.goal_pos,
            "obstacles": self.env.obstacles,  # Save all obstacle positions
            "num_obstacles": len(self.env.obstacles),
            "execution_time": execution_time,
            "total_steps": len(self.algorithm_steps),
            "timestamp": timestamp
        }
        
        # Save path if found
        for step in reversed(self.algorithm_steps):
            if "type" in step and step["type"] == "success" and "current_path" in step:
                summary["path_found"] = True
                summary["path_length"] = len(step["current_path"])
                summary["path"] = step["current_path"]  # Save the actual path
                break
        else:
            summary["path_found"] = False
        
        try:
            # Save to file
            with open(filename, 'w') as f:
                json.dump({
                    "summary": summary,
                    "steps": self.algorithm_steps
                }, f, indent=2)
                
            self.status_var.set(f"Execution completed. Results saved to {filename}")
        except Exception as e:
            self.status_var.set(f"Error saving results: {str(e)}")

    def generate_mosaic_visualization(self):
        """Generate a mosaic visualization of the planning process"""
        
        # Check if we have a successful path
        final_path = None
        for step in reversed(self.algorithm_steps):
            if "type" in step and step["type"] == "success" and "current_path" in step:
                final_path = step["current_path"]
                break
        
        if final_path is None:
            self.status_var.set("No path found, cannot create mosaic visualization")
            return
        
        # Create output directory if it doesn't exist
        images_dir = "output_images"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # Setup for visualization
        state = self.env.get_state()
        
        # Select key steps for visualization
        # We'll show: 1) Initial state, 2) Several intermediate steps, 3) Final path
        
        # First get all step indices where a path exists
        path_steps = []
        for i, step in enumerate(self.algorithm_steps):
            if "current_path" in step and step["current_path"]:
                path_steps.append(i)
        
        # Determine how many steps to show (max 15)
        num_steps_to_show = min(15, len(path_steps))
        
        # Evenly distribute the steps we'll show
        if num_steps_to_show <= 1:
            step_indices = path_steps  # Just show what we have
        else:
            # Always include the first step with a path and the last (success) step
            # Distribute the remaining evenly
            step_indices = [path_steps[0]]
            if num_steps_to_show > 2:
                # Calculate indices for intermediate steps
                step_size = (len(path_steps) - 1) // (num_steps_to_show - 1)
                for i in range(1, num_steps_to_show - 1):
                    idx = min(i * step_size, len(path_steps) - 1)
                    step_indices.append(path_steps[idx])
            # Add the final step
            if path_steps[-1] not in step_indices:
                step_indices.append(path_steps[-1])
        
        # Prepare data for visualization
        paths = []
        titles = []
        
        for idx in step_indices:
            step = self.algorithm_steps[idx]
            if "current_path" in step and step["current_path"]:
                paths.append(step["current_path"])
                
                # Create descriptive title
                if step["type"] == "success":
                    titles.append(f"Final Path (Step {idx})")
                elif idx == step_indices[0]:
                    titles.append(f"Initial Path (Step {idx})")
                else:
                    titles.append(f"Step {idx}")
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(images_dir, f"mosaic_grid_{timestamp}.png")
        
        # Call the visualization function with all steps
        self.visualize_mosaic(state, paths, self.grid_size, titles, save_path=filename)
        self.status_var.set(f"Mosaic visualization saved as {filename}")
        
        # # Also save individual visualizations for each important step
        # for i, (path, title) in enumerate(zip(paths, titles)):
        #     step_filename = os.path.join(images_dir, f"step_{timestamp}_{i:02d}.png")
        #     self.save_single_step_visualization(state, path, self.grid_size, title, save_path=step_filename)

    def save_single_step_visualization(self, environment, path, grid_size, title, save_path):
        """Save visualization of a single step"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        ax.set_title(title)
        
        cmap = plt.colormaps['tab10']
        for shape_id, points in environment["obstacle_shapes"].items():
            color = cmap(shape_id % 10)
            for obs in points:
                # Add label only for the first point of the first shape
                label = "Obstacle" if shape_id == 0 and obs == points[0] else ""
                ax.scatter(obs[1], obs[0], color=color, s=100, marker='s', label=label)
        
        # Plot agent
        ax.scatter(environment["agent"][1], environment["agent"][0], color='blue', s=150, marker='o', label="Start")
        
        # Plot goal
        ax.scatter(environment["goal"][1], environment["goal"][0], color='green', s=150, marker='*', label="Goal")
        
        # Plot path
        if path:
            path_x = [pos[1] for pos in path]
            path_y = [pos[0] for pos in path]
            ax.plot(path_x, path_y, color='orange', linewidth=2, label=f"Path (length: {len(path)})")
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Invert y-axis to match the main visualization
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def visualize_mosaic(self, environment, paths, grid_size, titles=None, save_path="mosaic_grid.png"):
        """Create a mosaic visualization with different paths"""
        num_paths = len(paths)
        
        # Determine grid layout - try to make it somewhat square
        cols = min(5, num_paths)
        rows = int(np.ceil(num_paths / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        # Handle case with single plot
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for idx in range(num_paths):
            ax = axes[idx]
            ax.set_xlim(-0.5, grid_size - 0.5)
            ax.set_ylim(-0.5, grid_size - 0.5)
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(0, grid_size, 1))
            ax.set_yticks(np.arange(0, grid_size, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
            
            # Set title
            if titles and idx < len(titles):
                ax.set_title(titles[idx])
            else:
                ax.set_title(f"Path {idx+1}")
            
            # Plot obstacles with shape colors
            cmap = plt.colormaps['tab10']
            for shape_id, points in environment["obstacle_shapes"].items():
                color = cmap(shape_id % 10)  # Added modulo for consistency
                for obs in points:
                    ax.scatter(obs[1], obs[0], color=color, s=80, marker='s')
            
            # Plot agent
            ax.scatter(environment["agent"][1], environment["agent"][0], color='blue', s=100, marker='o')
            
            # Plot goal
            ax.scatter(environment["goal"][1], environment["goal"][0], color='green', s=100, marker='*')
            
            # Plot path
            path = paths[idx]
            if path and path != [-1]:
                path_x = [pos[1] for pos in path]
                path_y = [pos[0] for pos in path]
                ax.plot(path_x, path_y, color='orange', linewidth=1.5)
            
            # Invert y-axis to match the main visualization
            ax.invert_yaxis()
        
        # Hide unused subplots
        for idx in range(num_paths, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def explain(self):
        """Generates LIME-based explanations for the current path planning problem"""
        self.status_var.set("Generating explanations... Please wait.")
        self.root.update()
        
        # Generate explanations with LIME
        importance = lime_explanation(self, num_samples=len(self.env.obstacle_shapes.keys()) + 10)
        
        if len(importance) == 0:
            self.status_var.set("No obstacles to explain.")
            return
        
        # Create a heatmap representation
        obstacle_keys = list(self.env.obstacle_shapes.keys())
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Map importance values to the grid
        for idx, shape_id in enumerate(obstacle_keys):
            imp_value = importance[idx]
            for pos in self.env.obstacle_shapes[shape_id]:
                x, y = pos
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[x, y] = imp_value
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use diverging colormap for positive/negative influences
        cmap = plt.cm.coolwarm
        
        # Handle potential extreme values
        grid_finite = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate bounds safely
        if np.all(grid_finite == 0):
            # All values are zero or NaN/inf (replaced with 0)
            vmin, vmax = -1, 1  # Use default range
        else:
            # Normalize around zero with a safe range
            max_abs = max(abs(np.min(grid_finite)), abs(np.max(grid_finite)))
            # Add a small buffer to avoid exactly zero range
            max_abs = max(max_abs, 0.1)
            vmin, vmax = -max_abs, max_abs
        
        # Create a safer normalization
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        # Create heatmap with explicit min/max to avoid hover errors
        heatmap = ax.imshow(grid_finite, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Disable hover tooltips that cause problems
        for artist in ax.get_children():
            artist.set_picker(None)
        
        # Draw grid lines for clarity
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # Mark start and goal positions
        if self.env.agent_pos:
            ax.scatter(self.env.agent_pos[1], self.env.agent_pos[0], 
                    color='blue', s=150, marker='o', label='Start')
        if self.env.goal_pos:
            ax.scatter(self.env.goal_pos[1], self.env.goal_pos[0], 
                    color='green', s=150, marker='*', label='Goal')
        
        # Add colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('Obstacle Importance')
        
        # Add title
        plt.title('Obstacle Importance for Path Planning\n'
                'Blue: Removal makes path worse (critical obstacle)\n'
                'Red: Removal helps path (obstructive obstacle)')
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        # Save explanation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        images_dir = "output_images"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        filepath = os.path.join(images_dir, f"explanation_{timestamp}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        
        # Also display it
        plt.show()
        
        self.status_var.set(f"Explanation generated and saved to {filepath}")

# LIME
def lime_explanation(app, num_samples=100):
    """
    Uses LIME technique to explain the importance of obstacles for path planning.
    """
    # Get obstacle keys
    obstacle_keys = list(app.env.obstacle_shapes.keys())
    num_obstacles = len(obstacle_keys)
    
    if num_obstacles == 0:
        return []
    
    # Generate all combinations for each obstacle being removed once
    combinations = app.env.generate_perturbation_combinations("remove_each_obstacle_once")
    
    # Also generate some random combinations (but limit total to avoid long processing)
    max_random = min(20, num_samples - len(combinations))
    random_combinations = []
    for _ in range(max_random):
        combo = [random.randint(0, 1) for _ in range(num_obstacles)]
        random_combinations.append(combo)
    
    # Combine all the combinations
    all_combinations = combinations + random_combinations
    
    X = []  # Perturbations (binary mask per sample)
    y = []  # Path costs per sample
    
    # Store original obstacles to restore at the end
    original_obstacles = app.env.obstacles.copy()
    
    # Create a progress counter for the status bar
    total_combinations = len(all_combinations)
    
    for i, combination in enumerate(all_combinations):
        # Update status
        app.status_var.set(f"Generating explanation: {i+1}/{total_combinations}")
        app.root.update()
        
        # Apply perturbation using the combination
        original_state, _ = app.env.generate_perturbation(combination=combination)
        
        # Run non-visual A* planning
        path = app.run_astar_for_analysis()
        
        # Measure outcome: path length or failure penalty
        if path:
            path_length = len(path)
            success = True
        else:
            path_length = app.grid_size * 2  # Penalty for no path
            success = False
        
        # Record results
        X.append(combination)
        y.append(path_length)
        
        # Restore environment to original state
        app.env.restore_from_perturbation(original_state)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Fit a Ridge regression model to explain obstacle importance
    explainer = Ridge(alpha=1.0)
    explainer.fit(X, y)
    
    # Get coefficients - positive means removing obstacle increases cost (important)
    # Negative means removing obstacle decreases cost (harmful for path planning)
    importance = explainer.coef_
    
    # Replace any infinite values with large finite values to avoid rendering errors
    importance = np.nan_to_num(importance, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    # Print results for debugging
    app.status_var.set("Explanation completed! Analyzing results...")
    app.root.update()
    
    for shape_id, imp in zip(obstacle_keys, importance):
        print(f"Obstacle shape #{shape_id}: importance {imp:.4f}")
        # Positive importance means removing this obstacle makes path worse (helpful)
        # Negative importance means removing this obstacle makes path better (obstructive)
    
    # Restore the original environment (just to be safe)
    app.env.obstacles = original_obstacles.copy()
    
    return importance

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = PathPlanningApp(root)
    root.mainloop()