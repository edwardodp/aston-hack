import numpy as np
from numba import njit
from . import constants as c

# --- 1. CONFIGURATION HELPERS ---

def get_personality_params(rowdiness_percent):
    """
    Returns a dictionary of physics coefficients based on rowdiness (0-100).
    (Kept in Python because it returns a Dictionary)
    """
    r = np.clip(rowdiness_percent / 100.0, 0.0, 1.0)
    
    def lerp(start, end, t):
        return start + t * (end - start)

    return {
        "desired_speed":   lerp(c.SPEED_CALM, c.SPEED_PANIC, r),
        "tau":             lerp(c.TAU_CALM, c.TAU_PANIC, r),
        "radius":          lerp(c.RADIUS_CALM, c.RADIUS_PANIC, r),
        "social_strength": lerp(c.SOCIAL_PUSH_CALM, c.SOCIAL_PUSH_PANIC, r),
        "noise":           lerp(c.NOISE_CALM, c.NOISE_PANIC, r)
    }

# --- SPATIAL PARTITIONING HELPER (The Secret Sauce) ---

@njit(cache=True)
def build_cell_list(pos, cell_size, grid_cols, grid_rows):
    """
    Builds a 'Head/Next' linked list for spatial partitioning.
    O(N) complexity.
    """
    N = len(pos)
    # Total number of cells in the spatial grid
    # We use a slightly coarser grid than the wall grid for efficiency
    # E.g., cell_size = interaction_radius (approx 30-40 pixels)
    
    num_cells = grid_cols * grid_rows
    
    # head[cell_index] -> points to the first agent index in that cell
    head = np.full(num_cells, -1, dtype=np.int32)
    
    # next_node[agent_index] -> points to the next agent index in the same cell
    next_node = np.full(N, -1, dtype=np.int32)
    
    for i in range(N):
        # Calculate cell index
        cx = int(pos[i, 0] / cell_size)
        cy = int(pos[i, 1] / cell_size)
        
        # Clamp to bounds
        if cx < 0: cx = 0
        if cx >= grid_cols: cx = grid_cols - 1
        if cy < 0: cy = 0
        if cy >= grid_rows: cy = grid_rows - 1
        
        cell_index = cy * grid_cols + cx
        
        # Insert at head (Linked List insertion)
        next_node[i] = head[cell_index]
        head[cell_index] = i
        
    return head, next_node

# --- 2. OPTIMIZED PHYSICS FORCES ---

@njit(cache=True)
def get_driving_force(pos, vel, goal, desired_speed, tau, mass_array):
    """
    Calculates self-driven force. (No changes needed here, purely local)
    """
    N = len(pos)
    force = np.zeros((N, 2))
    
    for i in range(N):
        # Vector to goal
        dx = goal[i, 0] - pos[i, 0]
        dy = goal[i, 1] - pos[i, 1]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq)
        
        if dist < 0.001: dist = 0.001
        
        # Desired Velocity
        ex = dx / dist
        ey = dy / dist
        
        des_vx = ex * desired_speed
        des_vy = ey * desired_speed
        
        # F = (v_desired - v_current) / tau * m
        fx = (des_vx - vel[i, 0]) / tau * mass_array[i]
        fy = (des_vy - vel[i, 1]) / tau * mass_array[i]
        
        force[i, 0] = fx
        force[i, 1] = fy
        
    return force

@njit(cache=True)
def get_social_force_optimized(pos, interaction_radius, repulsion_strength, head, next_node, cell_size, grid_cols, grid_rows):
    """
    Calculates Social Force using Spatial Grid.
    Complexity: O(N * density) instead of O(N^2)
    """
    N = len(pos)
    force = np.zeros((N, 2))
    
    # Precompute squared radius to avoid sqrt calls
    radius_sq = interaction_radius * interaction_radius
    
    for i in range(N):
        fx = 0.0
        fy = 0.0
        
        # Determine my cell
        cx = int(pos[i, 0] / cell_size)
        cy = int(pos[i, 1] / cell_size)
        
        # Check Neighbor Cells (3x3 grid)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx = cx + dx
                ny = cy + dy
                
                # Bounds check
                if nx >= 0 and nx < grid_cols and ny >= 0 and ny < grid_rows:
                    cell_idx = ny * grid_cols + nx
                    
                    # Traverse Linked List for this cell
                    j = head[cell_idx]
                    while j != -1:
                        if i != j:
                            # Calc distance
                            rx = pos[i, 0] - pos[j, 0]
                            ry = pos[i, 1] - pos[j, 1]
                            dist_sq = rx*rx + ry*ry
                            
                            if dist_sq < radius_sq and dist_sq > 0.0001:
                                dist = np.sqrt(dist_sq)
                                
                                # Normalized direction
                                ex = rx / dist
                                ey = ry / dist
                                
                                # Exponential Repulsion
                                mag = repulsion_strength * np.exp(-dist / (interaction_radius / 2.0))
                                
                                fx += ex * mag
                                fy += ey * mag
                        
                        j = next_node[j] # Move to next agent in cell
        
        force[i, 0] = fx
        force[i, 1] = fy
        
    return force

@njit(cache=True)
def get_contact_force_optimized(pos, diameters, stiffness, head, next_node, cell_size, grid_cols, grid_rows):
    """
    Calculates Contact Force using Spatial Grid.
    """
    N = len(pos)
    force = np.zeros((N, 2))
    
    for i in range(N):
        fx = 0.0
        fy = 0.0
        
        cx = int(pos[i, 0] / cell_size)
        cy = int(pos[i, 1] / cell_size)
        
        my_radius = diameters[i] / 2.0
        
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx = cx + dx
                ny = cy + dy
                
                if nx >= 0 and nx < grid_cols and ny >= 0 and ny < grid_rows:
                    cell_idx = ny * grid_cols + nx
                    
                    j = head[cell_idx]
                    while j != -1:
                        if i != j:
                            rx = pos[i, 0] - pos[j, 0]
                            ry = pos[i, 1] - pos[j, 1]
                            dist_sq = rx*rx + ry*ry
                            
                            radii_sum = my_radius + (diameters[j] / 2.0)
                            radii_sum_sq = radii_sum * radii_sum
                            
                            if dist_sq < radii_sum_sq and dist_sq > 0.0001:
                                dist = np.sqrt(dist_sq)
                                overlap = radii_sum - dist
                                
                                # Push direction
                                ex = rx / dist
                                ey = ry / dist
                                
                                mag = stiffness * overlap
                                
                                fx += ex * mag
                                fy += ey * mag
                                
                        j = next_node[j]
        
        force[i, 0] = fx
        force[i, 1] = fy
        
    return force

@njit(cache=True)
def _calc_force_magnitude(surface_dist, repulsion_strength, repulsion_range, wall_stiffness):
    contact_force = np.zeros_like(surface_dist)
    
    for i in range(len(surface_dist)):
        sd = surface_dist[i]
        if sd < 0:
            contact_force[i] = -sd * wall_stiffness
            
    # Soft force
    # eff_dist = max(0, surface_dist)
    eff_dist = np.maximum(0.0, surface_dist)
    soft_force = repulsion_strength * np.exp(-eff_dist / (repulsion_range / 2.0))
    
    return contact_force + soft_force

@njit(cache=True)
def get_grid_force(pos, grid_map, cell_w, cell_h, diameters, repulsion_strength, wall_stiffness, wall_repulsion_dist, grid_rows, grid_cols):
    """
    Optimized Grid Force calculation.
    We pass constants explicitly to avoid global lookup issues in Numba.
    """
    N = pos.shape[0]
    total_force = np.zeros((N, 2))
    
    # Compute indices
    col_indices = (pos[:, 0] / cell_w).astype(np.int64)
    row_indices = (pos[:, 1] / cell_h).astype(np.int64)
    
    # Iterate over neighbors
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            
            # Use arrays to store neighbor coords
            r_neigh = row_indices + dy
            c_neigh = col_indices + dx
            
            # Clip indices
            for i in range(N):
                if r_neigh[i] < 0: r_neigh[i] = 0
                if r_neigh[i] >= grid_rows: r_neigh[i] = grid_rows - 1
                if c_neigh[i] < 0: c_neigh[i] = 0
                if c_neigh[i] >= grid_cols: c_neigh[i] = grid_cols - 1
            
            # Check walls
            is_wall = np.zeros(N, dtype=np.bool_)
            for i in range(N):
                cell_val = grid_map[r_neigh[i], c_neigh[i]]
                # 1=WALL, 3=BARRIER, 2=POI (treated as wall for collisions?)
                # Original logic: WALL(1) or BARRIER(3) or POI(2)
                if cell_val == 1 or cell_val == 3 or cell_val == 2:
                    is_wall[i] = True
            
            if not np.any(is_wall): 
                continue
                
            cell_x1 = c_neigh * cell_w
            cell_y1 = r_neigh * cell_h
            # pass box as arrays
            
            # Numba can't pass "list of arrays" easily, so we modify _get_wall_interaction 
            # to calculate internally or we pass the box corners as arrays.
            # Let's modify logic slightly: calculate per-agent box.
            
            cell_box_x1 = cell_x1
            cell_box_y1 = cell_y1
            cell_box_x2 = cell_x1 + cell_w
            cell_box_y2 = cell_y1 + cell_h
            
            # INLINED _get_wall_interaction logic for arrays of boxes
            # (Because original function took a single static box, but here every agent has a different neighbor box)
            
            closest_x = np.empty(N)
            closest_y = np.empty(N)
            
            for i in range(N):
                # Clamp X
                px = pos[i, 0]
                if px < cell_box_x1[i]: closest_x[i] = cell_box_x1[i]
                elif px > cell_box_x2[i]: closest_x[i] = cell_box_x2[i]
                else: closest_x[i] = px
                
                # Clamp Y
                py = pos[i, 1]
                if py < cell_box_y1[i]: closest_y[i] = cell_box_y1[i]
                elif py > cell_box_y2[i]: closest_y[i] = cell_box_y2[i]
                else: closest_y[i] = py

            dist_vec_x = pos[:, 0] - closest_x
            dist_vec_y = pos[:, 1] - closest_y
            center_dist = np.sqrt(dist_vec_x**2 + dist_vec_y**2)
            
            # Degenerate check
            for i in range(N):
                if center_dist[i] < 0.001:
                    dist_vec_x[i] = 0.1
                    center_dist[i] = 0.1
            
            radius = diameters / 2.0
            surface_dist = center_dist - radius
            
            dir_x = dist_vec_x / center_dist
            dir_y = dist_vec_y / center_dist
            
            # Mag calc
            mag = _calc_force_magnitude(surface_dist, repulsion_strength, wall_repulsion_dist, wall_stiffness)
            
            # Apply
            for i in range(N):
                if is_wall[i] and surface_dist[i] < wall_repulsion_dist:
                    total_force[i, 0] += dir_x[i] * mag[i]
                    total_force[i, 1] += dir_y[i] * mag[i]
            
    return total_force


# --- 4. STATE MANAGEMENT ---

def init_state(num_agents, grid_map):
    # (Same as before - pure python initialization)
    cell_w = c.LOGICAL_WIDTH / c.GRID_COLS
    cell_h = c.LOGICAL_HEIGHT / c.GRID_ROWS
    
    poi_locations = []
    for r in range(c.GRID_ROWS):
        for col in range(c.GRID_COLS):
            if grid_map[r, col] == c.ID_POI:
                cx = (col * cell_w) + (cell_w / 2)
                cy = (r * cell_h) + (cell_h / 2)
                poi_locations.append([cx, cy])
    
    poi_array = np.array(poi_locations, dtype=np.float64) if poi_locations else np.empty((0, 2), dtype=np.float64)
    
    if len(poi_array) > 0:
        random_indices = np.random.randint(0, len(poi_array), size=num_agents)
        goals = poi_array[random_indices]
    else:
        center = np.array([c.LOGICAL_WIDTH / 2.0, c.LOGICAL_HEIGHT / 2.0])
        goals = np.full((num_agents, 2), center)

    valid_positions = []
    attempts = 0
    max_attempts = num_agents * 50
    walkable_mask = (grid_map == c.ID_NOTHING)

    while len(valid_positions) < num_agents and attempts < max_attempts:
        rx = np.random.uniform(0, c.LOGICAL_WIDTH)
        ry = np.random.uniform(0, c.LOGICAL_HEIGHT)
        c_idx = int(rx / cell_w)
        r_idx = int(ry / cell_h)
        if 0 <= r_idx < c.GRID_ROWS and 0 <= c_idx < c.GRID_COLS:
            if walkable_mask[r_idx, c_idx]:
                valid_positions.append([rx, ry])
        attempts += 1
        
    while len(valid_positions) < num_agents:
        valid_positions.append([c.LOGICAL_WIDTH/2, c.LOGICAL_HEIGHT/2])

    pos = np.array(valid_positions, dtype=np.float64)
    vel = np.zeros_like(pos)
    diameters = np.random.uniform(18.0, 28.0, size=num_agents)
    pressures = np.zeros(num_agents, dtype=np.float64)

    return {
        'pos': pos, 
        'vel': vel, 
        'diameters': diameters,
        'goals': goals,
        'pressures': pressures, 
        'grid': grid_map,
        'poi_locations': poi_array
    }

def tick(state, rowdiness_val, grid_map, poi_switch_chance=0.01):
    pos = state['pos']
    vel = state['vel']
    diameters = state['diameters']
    goals = state['goals']
    
    num_agents = len(pos)
    if num_agents == 0: return state
    
    # ... (Goal switching logic remains the same) ...
    poi_locs = state.get('poi_locations')
    if poi_switch_chance > 0 and poi_locs is not None and len(poi_locs) > 0:
        if np.random.rand() < 0.1: # Optimization: Only check switch logic occasionally
            switch_mask = np.random.rand(num_agents) < poi_switch_chance
            if np.sum(switch_mask) > 0:
                random_indices = np.random.randint(0, len(poi_locs), size=int(np.sum(switch_mask)))
                goals[switch_mask] = poi_locs[random_indices]

    params = get_personality_params(rowdiness_val * 100)
    mass_array = (diameters / c.REF_DIAMETER)**2 * c.BASE_MASS

    # --- NEW: Build Spatial Grid ---
    # Cell size should be roughly equal to the max interaction radius
    # 40.0 pixels is a safe upper bound for interaction_radius (panic mode is ~35)
    spatial_cell_size = 40.0 
    s_cols = int(c.LOGICAL_WIDTH / spatial_cell_size) + 1
    s_rows = int(c.LOGICAL_HEIGHT / spatial_cell_size) + 1
    
    head, next_node = build_cell_list(pos, spatial_cell_size, s_cols, s_rows)
    # -------------------------------

    # 1. Driving Force (Parallel)
    F_goal = get_driving_force(pos, vel, goals, params['desired_speed'], params['tau'], mass_array)
    
    # 2. Social Force (Spatial Optimized)
    F_social = get_social_force_optimized(
        pos, params['radius'], params['social_strength'], 
        head, next_node, spatial_cell_size, s_cols, s_rows
    )
    
    # 3. Contact Force (Spatial Optimized)
    F_contact = get_contact_force_optimized(
        pos, diameters, c.AGENT_CONTACT_STIFFNESS, 
        head, next_node, spatial_cell_size, s_cols, s_rows
    )
    
    # 4. Grid Force (Existing JIT)
    cell_w = c.LOGICAL_WIDTH / c.GRID_COLS
    cell_h = c.LOGICAL_HEIGHT / c.GRID_ROWS
    F_grid = get_grid_force(
        pos, grid_map, cell_w, cell_h, diameters, 
        c.WALL_REPULSION_STRENGTH, c.WALL_STIFFNESS, c.WALL_REPULSION_DIST,
        c.GRID_ROWS, c.GRID_COLS
    )
    
    F_noise = np.random.uniform(-1, 1, size=(num_agents, 2)) * params['noise']

    # Integration
    F_total = F_goal + F_social + F_contact + F_grid + F_noise
    
    acceleration = F_total / mass_array[:, np.newaxis]
    vel += acceleration * c.DT
    
    max_speed = params['desired_speed'] * 1.5
    speed_mag = np.linalg.norm(vel, axis=1)
    
    # Manual clipping loop is often faster in Numba, but numpy is fine here for now
    limit_mask = speed_mag > max_speed
    if np.any(limit_mask):
        vel[limit_mask] = (vel[limit_mask] / speed_mag[limit_mask, np.newaxis]) * max_speed
        
    pos += vel * c.DT
    
    # Boundary Clip
    pos[:, 0] = np.clip(pos[:, 0], 0, c.LOGICAL_WIDTH - 0.1)
    pos[:, 1] = np.clip(pos[:, 1], 0, c.LOGICAL_HEIGHT - 0.1)

    # Pressure Calculation (Reuse optimized forces)
    crush_forces = F_social + F_contact
    crush_mag = np.linalg.norm(crush_forces, axis=1)
    
    # (Barrier mask logic remains the same...)
    c_idx = (pos[:, 0] / cell_w).astype(int)
    r_idx = (pos[:, 1] / cell_h).astype(int)
    c_idx = np.clip(c_idx, 0, c.GRID_COLS-1)
    r_idx = np.clip(r_idx, 0, c.GRID_ROWS-1)
    
    near_barrier_mask = np.zeros(num_agents, dtype=bool)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny = np.clip(r_idx + dy, 0, c.GRID_ROWS-1)
            nx = np.clip(c_idx + dx, 0, c.GRID_COLS-1)
            near_barrier_mask |= (grid_map[ny, nx] == c.ID_BARRIER)

    relief_factor = np.ones(num_agents)
    relief_factor[near_barrier_mask] = 0.2 
    final_pressure = crush_mag * 0.10 * relief_factor
    
    state['pressures'] = np.clip(final_pressure, 0, 255)
    state['pos'] = pos
    state['vel'] = vel
    
    return state

# --- 5. GETTERS ---
def get_agents_pos(state): return state['pos']
def get_agents_pressure(state): return state['pressures']
def get_agents_size(state): return state['diameters']
def get_walls(state): return state['grid']

def get_average_pressure(state):
    """
    Returns the average pressure of all agents.
    """
    if len(state['pressures']) == 0:
        return 0.0
    return np.mean(state['pressures'])
