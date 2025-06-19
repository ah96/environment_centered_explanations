import random
import copy

class GridWorldEnv:
    def __init__(self, grid_size=10, num_obstacles=5, seed=0):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        random.seed(seed)
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
        
        successful_obstacles = 0
        
        # Generate obstacles without being limited by regions
        for shape_id in range(self.num_obstacles):
            # Try to find a valid location for this obstacle
            max_attempts = 100
            obstacle_created = False
            
            for attempt in range(max_attempts):
                # Pick a random starting point anywhere in the grid
                start_row = random.randint(0, self.grid_size - 1)
                start_col = random.randint(0, self.grid_size - 1)
                
                # Skip if this position is already occupied
                if [start_row, start_col] in self.obstacles:
                    continue
                
                # Generate a connected shape with random size (1-8 cells)
                shape_size = random.randint(1, 8)
                shape_points = [[start_row, start_col]]
                
                # Add connected points
                for _ in range(shape_size - 1):
                    if not shape_points:
                        break
                        
                    # Get a random point from existing shape to expand from
                    base_point = random.choice(shape_points)
                    
                    # Try neighbors
                    neighbors = [
                        [base_point[0]-1, base_point[1]],  # up
                        [base_point[0]+1, base_point[1]],  # down
                        [base_point[0], base_point[1]-1],  # left
                        [base_point[0], base_point[1]+1]   # right
                    ]
                    
                    # Filter valid neighbors
                    valid_neighbors = [
                        n for n in neighbors 
                        if 0 <= n[0] < self.grid_size and 
                        0 <= n[1] < self.grid_size and 
                        n not in shape_points and
                        n not in self.obstacles
                    ]
                    
                    if valid_neighbors:
                        new_point = random.choice(valid_neighbors)
                        shape_points.append(new_point)
                
                # Only accept if we have at least one point and it's not conflicting
                if shape_points and all(p not in self.obstacles for p in shape_points):
                    # Store the shape using the current successful_obstacles count as ID
                    self.obstacle_shapes[successful_obstacles] = shape_points
                    # Add all points to obstacles list
                    self.obstacles.extend(shape_points)
                    successful_obstacles += 1
                    obstacle_created = True
                    break
            
            if not obstacle_created:
                # Try to place a single-cell obstacle at any free position
                free_positions = [
                    [r, c] for r in range(self.grid_size) 
                    for c in range(self.grid_size)
                    if [r, c] not in self.obstacles
                ]
                
                if free_positions:
                    pos = random.choice(free_positions)
                    # Use the current successful_obstacles count as ID
                    self.obstacle_shapes[successful_obstacles] = [pos]
                    self.obstacles.append(pos)
                    successful_obstacles += 1
        
        # Print warning if we couldn't generate all requested obstacles
        if successful_obstacles < self.num_obstacles:
            print(f"WARNING: Could only generate {successful_obstacles}/{self.num_obstacles} obstacles "
                f"for grid size {self.grid_size}x{self.grid_size}. Grid may be too small or too crowded.")
        
        # Update num_obstacles to reflect actual number generated
        self.num_obstacles = successful_obstacles

        
    # def generate_obstacles(self):
    #     self.obstacle_shapes = {}
    #     self.obstacles = []
        
    #     # Divide grid into regions for obstacles
    #     regions = self.divide_grid_into_regions()
        
    #     # Generate an obstacle shape in each region
    #     for shape_id, region in enumerate(regions):
    #         if shape_id >= self.num_obstacles:
    #             break
                
    #         # Generate a connected shape within this region
    #         shape_points = self.generate_connected_shape(region)
            
    #         # Store the shape with its ID
    #         self.obstacle_shapes[shape_id] = shape_points
            
    #         # Add all points to obstacles list for quick lookup
    #         self.obstacles.extend(shape_points)
    
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

    def generate_perturbation(self, strategy="random", combination=None, mode="remove"):
            """
            Generates and applies perturbation by removing obstacle shapes based on strategy and combination.
            
            Args:
                strategy (str): The perturbation strategy.
                combination (list, optional): A specific combination to apply (list of 0s and 1s).
                    If provided, this overrides the strategy.
                mode (str): The perturbation mode ("remove", "move", "random", "minimal_move").
                    
            Returns:
                tuple: (original_state, shapes_removed_ids)
                    original_state (dict): A dictionary with copies of obstacles and obstacle_shapes.
                    shapes_removed_ids (list): A list of the IDs of the obstacle shapes that were removed.
            """
            # Create deep copies of both obstacles and obstacle_shapes
            original_state = {
                'obstacles': copy.deepcopy(self.obstacles),
                'obstacle_shapes': copy.deepcopy(self.obstacle_shapes)
            }
            
            # Get shape IDs at the start of the function to maintain consistency
            all_shape_ids = list(self.obstacle_shapes.keys())
            shapes_to_remove_ids = []
            
            if not all_shape_ids:  # No obstacles to remove
                return original_state, []
            
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
            
            # Apply the perturbation by removing or moving the selected obstacle shapes
            for shape_id in shapes_to_remove_ids:
                if mode == "remove":
                    self.remove_obstacle_shape(shape_id)
                elif mode == "move":
                    self.move_obstacle_shape(shape_id)
                elif mode == "random":
                    if random.random() < 0.5:
                        self.remove_obstacle_shape(shape_id)
                    else:
                        self.move_obstacle_shape(shape_id)
                elif mode == "minimal_move":
                    self.move_obstacle_shape_min_displacement(shape_id)
                
            return original_state, shapes_to_remove_ids

    def restore_from_perturbation(self, original_state):
            """Restore the environment to the state before perturbation
            
            Args:
                original_state (dict): A dictionary containing original obstacles and obstacle_shapes.
            """
            # Restore both obstacles and obstacle_shapes
            self.obstacles = original_state['obstacles'].copy()
            self.obstacle_shapes = original_state['obstacle_shapes'].copy()

    def get_state(self):
        return {
            "grid_size": self.grid_size,
            "agent": self.agent_pos,
            "goal": self.goal_pos,
            "obstacles": self.obstacles,
            "obstacle_shapes": self.obstacle_shapes
        }
    
    def move_obstacle_shape(self, shape_id, max_attempts=10):
        """Attempt to move the obstacle shape to a new valid location."""
        if shape_id not in self.obstacle_shapes:
            return False
        
        old_shape = self.obstacle_shapes[shape_id]
        shape_vector = [(r - old_shape[0][0], c - old_shape[0][1]) for r, c in old_shape]

        for _ in range(max_attempts):
            new_anchor = random.choice([
                [r, c]
                for r in range(self.grid_size)
                for c in range(self.grid_size)
                if [r, c] not in self.obstacles
            ])
            new_shape = [[new_anchor[0] + dr, new_anchor[1] + dc] for dr, dc in shape_vector]
            if all(0 <= r < self.grid_size and 0 <= c < self.grid_size and [r, c] not in self.obstacles for r, c in new_shape):
                for point in old_shape:
                    if point in self.obstacles:
                        self.obstacles.remove(point)
                self.obstacle_shapes[shape_id] = new_shape
                self.obstacles.extend(new_shape)
                return True
        return False
    
    def move_obstacle_shape_min_displacement(self, shape_id, max_radius=3):
        """
        Move an obstacle shape to a nearby valid location with minimal displacement.
        
        Args:
            shape_id (int): ID of the obstacle shape to move
            max_radius (int): Max Manhattan distance for displacement
            
        Returns:
            bool: True if successfully moved, False otherwise
        """
        import random

        if shape_id not in self.obstacle_shapes:
            return False

        old_shape = self.obstacle_shapes[shape_id]
        shape_vector = [(r - old_shape[0][0], c - old_shape[0][1]) for r, c in old_shape]

        # Try small displacements
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # up, down, left, right
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonals
        ]
        random.shuffle(directions)

        for radius in range(1, max_radius + 1):
            for dr, dc in directions:
                delta = (dr * radius, dc * radius)
                new_anchor = [old_shape[0][0] + delta[0], old_shape[0][1] + delta[1]]
                new_shape = [[new_anchor[0] + d_r, new_anchor[1] + d_c] for d_r, d_c in shape_vector]

                if all(
                    0 <= r < self.grid_size and 0 <= c < self.grid_size and [r, c] not in self.obstacles
                    for r, c in new_shape
                ):
                    # Remove old shape points
                    for p in old_shape:
                        if p in self.obstacles:
                            self.obstacles.remove(p)

                    # Apply new shape
                    self.obstacle_shapes[shape_id] = new_shape
                    self.obstacles.extend(new_shape)
                    return True
        return False

    def clone(self):
        new_env = GridWorldEnv(grid_size=self.grid_size)
        new_env.agent_pos = self.agent_pos[:]
        new_env.goal_pos = self.goal_pos[:]
        new_env.obstacles = [tuple(o) for o in self.obstacles]  # ensure deep copy
        new_env.obstacle_shapes = {
            sid: [tuple(cell) for cell in shape]
            for sid, shape in self.obstacle_shapes.items()
        }
        return new_env

    def remove_obstacle_shape(self, shape_id):
        """Remove an obstacle shape and update obstacle list."""
        if shape_id in self.obstacle_shapes:
            for pt in self.obstacle_shapes[shape_id]:
                if pt in self.obstacles:
                    self.obstacles.remove(pt)
            del self.obstacle_shapes[shape_id]

    def add_obstacle_shape(self, shape_cells):
        """Adds a list of (x, y) cells as a new shape to the environment."""
        sid = max(self.obstacle_shapes.keys(), default=0) + 1
        self.obstacle_shapes[sid] = shape_cells
        for cell in shape_cells:
            if cell not in self.obstacles:
                self.obstacles.append(cell)
        return sid

    def generate_random_position(self):
        """Generate a random free (non-obstacle) grid cell."""
        import random
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if [x, y] not in self.obstacles:
                return [x, y]

    def update_obstacles_from_shapes(self):
        """Regenerate flat obstacle list from obstacle_shapes."""
        self.obstacles = [cell for shape in self.obstacle_shapes.values() for cell in shape]
