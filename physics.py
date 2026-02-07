import numpy as np
import math
import constants as c

# --- GLOBAL STATE ---
positions = np.empty((0, 2))
velocities = np.empty((0, 2))
# NEW: Store individual agent diameters
agent_diameters = np.empty((0,)) 
goals = np.array([c.LOGICAL_WIDTH / 2, c.LOGICAL_HEIGHT / 2]) 
num_agents = 0

# --- GRID STATE ---
grid_map = None 
cell_w = 0.0
cell_h = 0.0

# --- HELPER FUNCTIONS ---

def map_val(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)

def get_driving_force(pos, vel, goal, desired_speed, tau):
    direction_vector = goal - pos 
    distance = np.linalg.norm(direction_vector, axis=1)
    distance[distance == 0] = 0.001 
    
    unit_direction = direction_vector / distance[:, np.newaxis]
    desired_velocity = unit_direction * desired_speed
    force = (desired_velocity - vel) / tau
    return force

def get_social_force(pos, interaction_radius, repulsion_strength):
    N = pos.shape[0]
    if N == 0: return np.zeros((0, 2))

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    directions = diff / (dists[:, :, np.newaxis] + 1e-8)
    
    mask = 1 - np.eye(N)
    interaction_mask = dists < interaction_radius
    active_mask = mask * interaction_mask
    
    magnitude = repulsion_strength * np.exp(-dists / (interaction_radius/2))
    force_vectors = directions * magnitude[:, :, np.newaxis] * active_mask[:, :, np.newaxis]
    return np.sum(force_vectors, axis=1)

def get_grid_force(pos, grid, cell_w, cell_h, agent_diameter_arr, repulsion_strength=5000.0, repulsion_range=30.0):
    """
    Calculates repulsion from grid walls taking Radius into account.
    """
    N = pos.shape[0]
    total_force = np.zeros_like(pos)
    
    if grid is None:
        return total_force

    # 1. Determine which cell each agent is in
    col_indices = (pos[:, 0] / cell_w).astype(int)
    row_indices = (pos[:, 1] / cell_h).astype(int)

    # 2. Iterate through 3x3 neighborhood
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            
            r_neigh = row_indices + dy
            c_neigh = col_indices + dx
            
            # Check bounds and if it is a wall
            # Clip indices for lookup safety (mask handles logic)
            safe_r = np.clip(r_neigh, 0, c.GRID_ROWS - 1)
            safe_c = np.clip(c_neigh, 0, c.GRID_COLS - 1)
            
            valid_mask = (r_neigh >= 0) & (r_neigh < c.GRID_ROWS) & \
                         (c_neigh >= 0) & (c_neigh < c.GRID_COLS)
            
            is_wall = (grid[safe_r, safe_c] == c.ID_WALL) | (grid[safe_r, safe_c] == c.ID_BARRIER)
            active_mask = valid_mask & is_wall
            
            if not np.any(active_mask):
                continue

            # --- CLOSEST POINT CALCULATION ---
            cell_x1 = c_neigh * cell_w
            cell_y1 = r_neigh * cell_h
            cell_x2 = cell_x1 + cell_w
            cell_y2 = cell_y1 + cell_h
            
            # Find closest point on the wall box to the agent center
            closest_x = np.maximum(cell_x1, np.minimum(pos[:, 0], cell_x2))
            closest_y = np.maximum(cell_y1, np.minimum(pos[:, 1], cell_y2))
            closest_points = np.stack([closest_x, closest_y], axis=1)
            
            # Vector from Wall -> Agent Center
            dist_vec = pos - closest_points
            center_dist = np.linalg.norm(dist_vec, axis=1) 
            
            # --- RADIUS CORRECTION ---
            # Surface Distance = Center Distance - Radius
            radius = agent_diameter_arr / 2.0
            surface_dist = center_dist - radius
            
            # 1. HANDLE "INSIDE WALL" or "OVERLAP"
            # If center_dist ~ 0 (inside cell) OR surface_dist < 0 (overlapping edge)
            
            # Fix degenerate case (center inside wall)
            degenerate_mask = center_dist < 0.001
            if np.any(degenerate_mask):
                cell_cx = cell_x1 + cell_w/2
                cell_cy = cell_y1 + cell_h/2
                push_dir = pos - np.stack([cell_cx, cell_cy], axis=1)
                push_len = np.linalg.norm(push_dir, axis=1)
                
                # Push out from center
                dist_vec[degenerate_mask] = (push_dir / (push_len[:,np.newaxis]+1e-8))[degenerate_mask]
                center_dist[degenerate_mask] = 0.001 # Avoid 0 division below

            # Normalize direction (Wall -> Agent)
            direction = dist_vec / (center_dist[:, np.newaxis] + 1e-8)
            
            # 2. CALCULATE FORCE MAGNITUDE
            # Combine Spring Force (Overlap) and Exponential Force (Proximity)
            
            # A. Spring Force (Hooke's Law) for overlap
            contact_force = np.zeros_like(surface_dist)
            overlap_mask = surface_dist < 0
            # k * overlap. k=1000 is stiff.
            contact_force[overlap_mask] = -surface_dist[overlap_mask] * 1000.0 
            
            # B. Soft Repulsion (Exponential)
            # Clamp distance to 0 for exp calculation to avoid explosion
            eff_dist = np.maximum(0, surface_dist)
            soft_force = repulsion_strength * np.exp(-eff_dist / (repulsion_range/2))
            
            total_mag = contact_force + soft_force
            
            # Mask checks
            range_mask = surface_dist < repulsion_range
            final_mask = active_mask & range_mask
            
            force = direction * total_mag[:, np.newaxis] * final_mask[:, np.newaxis]
            total_force += force

    return total_force

# --- CORE EXPOSED FUNCTIONS ---

def init_simulation(n_agents_to_spawn, grid_array):
    """
    Initializes the simulation state based on the provided 32x32 grid.
    """
    global positions, velocities, goals, num_agents
    global grid_map, cell_w, cell_h
    global agent_diameters # Access global array
    
    # Store Grid Global State
    grid_map = grid_array
    num_agents = n_agents_to_spawn
    
    # Calculate cell dimensions
    cell_w = c.LOGICAL_WIDTH / c.GRID_COLS
    cell_h = c.LOGICAL_HEIGHT / c.GRID_ROWS
    
    poi_locations = []

    # 1. Parse Grid (Only needed for POIs now)
    for row in range(c.GRID_ROWS):
        for col in range(c.GRID_COLS):
            val = grid_array[row, col]
            if val == c.ID_POI:
                x1 = col * cell_w
                y1 = row * cell_h
                cx = x1 + (cell_w / 2)
                cy = y1 + (cell_h / 2)
                poi_locations.append([cx, cy])

    # 2. Set Goal
    if poi_locations:
        goals = np.mean(poi_locations, axis=0)
    else:
        goals = np.array([c.LOGICAL_WIDTH/2, c.LOGICAL_HEIGHT/2])

    # 3. Spawn Agents
    valid_positions = []
    attempts = 0
    max_attempts = n_agents_to_spawn * 100
    
    while len(valid_positions) < n_agents_to_spawn and attempts < max_attempts:
        rx = np.random.uniform(0, c.LOGICAL_WIDTH)
        ry = np.random.uniform(0, c.LOGICAL_HEIGHT)
        
        col_idx = int(rx / cell_w)
        row_idx = int(ry / cell_h)
        
        if 0 <= row_idx < c.GRID_ROWS and 0 <= col_idx < c.GRID_COLS:
            if grid_array[row_idx, col_idx] == c.ID_NOTHING:
                valid_positions.append([rx, ry])
        
        attempts += 1
        
    positions = np.array(valid_positions)
    num_agents = len(positions)
    velocities = np.zeros((num_agents, 2))
    
    # NEW: Initialize random diameters [18, 28]
    agent_diameters = np.random.randint(18, 28, size=num_agents)


def tick(rowdiness_val):
    """
    Advances the physics simulation by one step (DT).
    """
    global positions, velocities
    
    if num_agents == 0:
        return positions

    # 1. Map Parameters
    p_desired_speed = map_val(rowdiness_val, 0, 100, 80.0, 400.0) 
    p_tau = map_val(rowdiness_val, 0, 100, 0.8, 0.1)
    p_radius = map_val(rowdiness_val, 0, 100, 100.0, 20.0)
    p_social_strength = map_val(rowdiness_val, 0, 100, 50.0, 1000.0)
    p_noise = map_val(rowdiness_val, 0, 100, 0.0, 500.0)

    # 2. Calculate Forces
    F_goal = get_driving_force(positions, velocities, goals, desired_speed=p_desired_speed, tau=p_tau)
    F_social = get_social_force(positions, interaction_radius=p_radius, repulsion_strength=p_social_strength)
    
    # Grid Force: Now using Agent Radius    
    F_grid = get_grid_force(positions, grid_map, cell_w, cell_h, 
                            agent_diameter_arr=agent_diameters, 
                            repulsion_strength=1000.0, 
                            repulsion_range=1.0)
    
    F_noise = np.random.uniform(-1, 1, size=(num_agents, 2)) * p_noise

    # 3. Integration
    F_total = F_goal + F_social + F_grid + F_noise
    acceleration = F_total / 3.0 

    velocities += acceleration * c.DT
    
    # Speed Limit
    max_speed = p_desired_speed * 1.5
    speed_mag = np.linalg.norm(velocities, axis=1)
    mask = speed_mag > max_speed
    velocities[mask] = (velocities[mask] / speed_mag[mask, np.newaxis]) * max_speed
    
    positions += velocities * c.DT

    # 4. Boundary Clamp (Keep inside world)
    positions[:, 0] = np.clip(positions[:, 0], 0, c.LOGICAL_WIDTH)
    positions[:, 1] = np.clip(positions[:, 1], 0, c.LOGICAL_HEIGHT)
    
    return positions

# --- GETTERS ---
def get_agents_pos():
    return positions

def get_agents_pressure():
    # Placeholder
    return np.zeros(num_agents, dtype=np.int32)

def get_agents_size():
    # Returns the actual per-agent diameters
    return agent_diameters
    
def get_walls():
    return grid_map