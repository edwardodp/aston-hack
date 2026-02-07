import numpy as np
import math
import constants as c

# --- GLOBAL STATE ---
# These hold the data for the current simulation
positions = np.empty((0, 2))
velocities = np.empty((0, 2))
goals = np.array([c.LOGICAL_WIDTH / 2, c.LOGICAL_HEIGHT / 2]) # Default center
walls = [] # List of [x1, y1, x2, y2]
num_agents = 0

# --- HELPER FUNCTIONS ---

def map_val(val, in_min, in_max, out_min, out_max):
    """Linearly maps a value from one range to another."""
    return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)

def get_driving_force(pos, vel, goal, desired_speed, tau):
    """Calculates the force pulling agents toward the goal."""
    direction_vector = goal - pos 
    distance = np.linalg.norm(direction_vector, axis=1)
    
    # Avoid division by zero
    distance[distance == 0] = 0.001 
    
    unit_direction = direction_vector / distance[:, np.newaxis]
    desired_velocity = unit_direction * desired_speed
    force = (desired_velocity - vel) / tau
    return force

def get_social_force(pos, interaction_radius, repulsion_strength):
    """Calculates the repulsive force between all pairs of agents."""
    N = pos.shape[0]
    if N == 0: return np.zeros((0, 2))

    # 1. Diff vectors: diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # 2. Directions
    directions = diff / (dists[:, :, np.newaxis] + 1e-8)
    
    # 3. Magnitude (Exponential decay)
    # Mask out self-interaction and distant agents
    mask = 1 - np.eye(N)
    interaction_mask = dists < interaction_radius
    active_mask = mask * interaction_mask
    
    magnitude = repulsion_strength * np.exp(-dists / (interaction_radius/2))
    
    # 4. Sum forces
    force_vectors = directions * magnitude[:, :, np.newaxis] * active_mask[:, :, np.newaxis]
    return np.sum(force_vectors, axis=1)

def get_wall_force(pos, walls, repulsion_strength=200.0, repulsion_range=50.0):
    """Calculates repulsion from a list of line segment walls."""
    total_force = np.zeros_like(pos)
    if not walls:
        return total_force

    for wall in walls:
        start = np.array([wall[0], wall[1]])
        end = np.array([wall[2], wall[3]])
        
        wall_vec = end - start
        wall_len_sq = np.dot(wall_vec, wall_vec)
        if wall_len_sq == 0: continue

        agent_vec = pos - start
        
        # Project agent onto wall line segment
        t = np.sum(agent_vec * wall_vec, axis=1) / wall_len_sq
        t = np.clip(t, 0, 1)
        
        closest_point = start + (t[:, np.newaxis] * wall_vec)
        dist_vec = pos - closest_point
        dist = np.linalg.norm(dist_vec, axis=1)
        
        # Avoid division by zero
        direction = dist_vec / (dist[:, np.newaxis] + 1e-8)
        
        active_mask = dist < repulsion_range
        magnitude = repulsion_strength * np.exp(-dist / (repulsion_range/2))
        
        force = direction * magnitude[:, np.newaxis] * active_mask[:, np.newaxis]
        total_force += force
        
    return total_force

# --- CORE EXPOSED FUNCTIONS ---

def init_simulation(n_agents_to_spawn, grid_array):
    """
    Initializes the simulation state based on the provided 32x32 grid.
    
    Args:
        n_agents_to_spawn (int): Number of agents.
        grid_array (np.array): 32x32 numpy array of uint8.
                               0=Empty, 1=Wall, 2=POI, 3=Barrier
    """
    global positions, velocities, goals, walls, num_agents
    
    num_agents = n_agents_to_spawn
    walls = []
    poi_locations = []
    
    cell_w = c.LOGICAL_WIDTH / c.GRID_COLS
    cell_h = c.LOGICAL_HEIGHT / c.GRID_ROWS

    # 1. Parse Grid
    # We iterate over the grid to find Walls and POIs
    for row in range(c.GRID_ROWS):
        for col in range(c.GRID_COLS):
            val = grid_array[row, col]
            
            # Calculate physical bounds of this cell
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            
            if val == c.ID_WALL or val == c.ID_BARRIER:
                # Create 4 wall segments for the cell (bounding box)
                # Optimization: In a real game, you'd merge these, but this works for physics
                walls.append([x1, y1, x2, y1]) # Top
                walls.append([x2, y1, x2, y2]) # Right
                walls.append([x2, y2, x1, y2]) # Bottom
                walls.append([x1, y2, x1, y1]) # Left
                
            elif val == c.ID_POI:
                # Center of the cell
                cx = x1 + (cell_w / 2)
                cy = y1 + (cell_h / 2)
                poi_locations.append([cx, cy])

    # 2. Set Goal
    # If POIs exist, set the goal to the average location of all POI cells
    if poi_locations:
        goals = np.mean(poi_locations, axis=0)
    else:
        # Default fallback: Center of world
        goals = np.array([c.LOGICAL_WIDTH/2, c.LOGICAL_HEIGHT/2])

    # 3. Spawn Agents
    # Randomly place agents in valid (empty) space
    # Simple retry logic to avoid spawning inside walls
    valid_positions = []
    attempts = 0
    max_attempts = n_agents_to_spawn * 100
    
    while len(valid_positions) < n_agents_to_spawn and attempts < max_attempts:
        rx = np.random.uniform(0, c.LOGICAL_WIDTH)
        ry = np.random.uniform(0, c.LOGICAL_HEIGHT)
        
        # Check grid index
        col_idx = int(rx / cell_w)
        row_idx = int(ry / cell_h)
        
        # Boundary check
        if 0 <= row_idx < c.GRID_ROWS and 0 <= col_idx < c.GRID_COLS:
            # Only spawn if space is empty (ID_NOTHING)
            if grid_array[row_idx, col_idx] == c.ID_NOTHING:
                valid_positions.append([rx, ry])
        
        attempts += 1
        
    positions = np.array(valid_positions)
    
    # If we couldn't find enough spots, truncate
    num_agents = len(positions)
    velocities = np.zeros((num_agents, 2))


def tick(rowdiness_val):
    """
    Advances the physics simulation by one step (DT).
    Controlled by the external Game Loop.
    
    Args:
        rowdiness_val (float): 0.0 to 100.0
    """
    global positions, velocities
    
    if num_agents == 0:
        return

    # 1. Map Parameters (Scaled for 1024.0 world size)
    # Note: Previous speeds were ~1-5 for a 10.0 world. 
    # Now world is 100x bigger, so speeds must be ~100x bigger.
    
    # Speed: 80 px/s (Walk) to 400 px/s (Run)
    p_desired_speed = map_val(rowdiness_val, 0, 100, 80.0, 400.0) 
    
    # Tau: 0.8s (Lazy) to 0.1s (Aggressive)
    p_tau = map_val(rowdiness_val, 0, 100, 0.8, 0.1)
    
    # Radius: 100px (Polite) to 20px (Pushy)
    p_radius = map_val(rowdiness_val, 0, 100, 100.0, 20.0)
    
    # Strength: 2000 to 5000 (Scaled up for larger distances)
    p_social_strength = map_val(rowdiness_val, 0, 100, 2000.0, 5000.0)
    
    # Noise: 0 to 500
    p_noise = map_val(rowdiness_val, 0, 100, 0.0, 500.0)

    # 2. Calculate Forces
    F_goal = get_driving_force(positions, velocities, goals, desired_speed=p_desired_speed, tau=p_tau)
    F_social = get_social_force(positions, interaction_radius=p_radius, repulsion_strength=p_social_strength)
    F_wall = get_wall_force(positions, walls, repulsion_strength=5000.0, repulsion_range=30.0)
    F_noise = np.random.uniform(-1, 1, size=(num_agents, 2)) * p_noise

    # 3. Integration
    F_total = F_goal + F_social + F_wall + F_noise
    acceleration = F_total / 3.0  # Mass = 3

    velocities += acceleration * c.DT
    
    # Speed Limit
    max_speed = p_desired_speed * 1.5
    speed_mag = np.linalg.norm(velocities, axis=1)
    mask = speed_mag > max_speed
    velocities[mask] = (velocities[mask] / speed_mag[mask, np.newaxis]) * max_speed
    
    positions += velocities * c.DT

    # 4. Simple Boundary Clamp (Keep inside world)
    positions[:, 0] = np.clip(positions[:, 0], 0, c.LOGICAL_WIDTH)
    positions[:, 1] = np.clip(positions[:, 1], 0, c.LOGICAL_HEIGHT)
    
    # Return positions for the view to render (optional, but helpful)
    return positions