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

# --- 2. PHYSICS FORCES (JIT COMPILED) ---

@njit(cache=True)
def get_driving_force(pos, vel, goal, desired_speed, tau, mass_array):
    """
    Calculates self-driven force. 
    """
    direction_vector = goal - pos 
    # Numba supports linalg.norm
    distance = np.empty(len(direction_vector), dtype=np.float64)
    for i in range(len(direction_vector)):
        val = np.linalg.norm(direction_vector[i])
        distance[i] = val if val > 0 else 0.001
    
    # We reshape to allow broadcasting (N, 1)
    unit_direction = direction_vector / distance.reshape(-1, 1)
    desired_velocity = unit_direction * desired_speed
    
    force = (desired_velocity - vel) / tau
    return force * mass_array.reshape(-1, 1)

@njit(cache=True)
def get_social_force(pos, interaction_radius, repulsion_strength):
    """
    Psychological Repulsion (Exponential).
    """
    N = pos.shape[0]
    if N == 0: return np.zeros((0, 2))

    # Numba handles broadcasting, but explicit loops are often faster and use less memory.
    # However, for simplicity/compatibility, we stick to the array syntax which Numba optimizes well.
    diff = pos.reshape(N, 1, 2) - pos.reshape(1, N, 2)
    
    # Manually compute norm to avoid complex axis arguments if needed, 
    # but modern Numba supports axis=2.
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero
    # We add epsilon to dists where dists is 0
    safe_dists = dists.copy()
    for i in range(N):
        for j in range(N):
            if safe_dists[i, j] < 1e-8:
                safe_dists[i, j] = 1.0 # arbitrary, we will mask it out anyway

    directions = diff / safe_dists.reshape(N, N, 1)
    
    # Identity mask
    not_self_mask = np.ones((N, N)) - np.eye(N)
    range_mask = dists < interaction_radius
    active_mask = not_self_mask * range_mask
    
    # Exponential decay
    magnitude = repulsion_strength * np.exp(-dists / (interaction_radius / 2.0))
    
    # Combine
    force_vectors = directions * magnitude.reshape(N, N, 1) * active_mask.reshape(N, N, 1)
    
    return np.sum(force_vectors, axis=1)

@njit(cache=True)
def get_agent_contact_force(pos, diameters, stiffness):
    """
    Physical Contact Force (Hooke's Law).
    Strictly handles body overlap.
    """
    N = pos.shape[0]
    if N == 0: return np.zeros((0, 2))
    
    diff = pos.reshape(N, 1, 2) - pos.reshape(1, N, 2)
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    radii = diameters / 2.0
    radii_sum = radii.reshape(N, 1) + radii.reshape(1, N)
    
    overlap = radii_sum - dists
    overlap_mask = overlap > 0
    
    not_self_mask = np.ones((N, N)) - np.eye(N)
    # Bitwise AND on boolean arrays
    active_mask = overlap_mask & (not_self_mask > 0.5)
    
    # Safe division
    safe_dists = dists.copy()
    for i in range(N):
        for j in range(N):
            if safe_dists[i, j] < 1e-8:
                safe_dists[i, j] = 1.0 

    directions = diff / safe_dists.reshape(N, N, 1)
    magnitude = stiffness * overlap
    
    force_vectors = directions * magnitude.reshape(N, N, 1) * active_mask.reshape(N, N, 1)
    
    return np.sum(force_vectors, axis=1)

# --- 3. MODULAR GRID FORCE (JIT COMPILED) ---

@njit(cache=True)
def _get_wall_interaction(pos, diameters, cell_box):
    # Numba requires explicit array creation for stacking if shapes differ,
    # but here we compute vector-wise
    
    # pos is (N, 2), cell_box is (4,) [x1, y1, x2, y2]
    # We clamp pos x/y to the box edges
    
    # Manual clipping for Numba speed
    closest_x = np.empty_like(pos[:, 0])
    closest_y = np.empty_like(pos[:, 1])
    
    for i in range(len(pos)):
        px = pos[i, 0]
        py = pos[i, 1]
        
        # Clip X
        if px < cell_box[0]: cx = cell_box[0]
        elif px > cell_box[2]: cx = cell_box[2]
        else: cx = px
            
        # Clip Y
        if py < cell_box[1]: cy = cell_box[1]
        elif py > cell_box[3]: cy = cell_box[3]
        else: cy = py
            
        closest_x[i] = cx
        closest_y[i] = cy

    dist_vec_x = pos[:, 0] - closest_x
    dist_vec_y = pos[:, 1] - closest_y
    
    # Recombine
    center_dist = np.sqrt(dist_vec_x**2 + dist_vec_y**2)

    # Handle degenerates
    for i in range(len(center_dist)):
        if center_dist[i] < 0.001:
            dist_vec_x[i] = 0.1
            dist_vec_y[i] = 0.0
            center_dist[i] = 0.1

    radius = diameters / 2.0
    surface_dist = center_dist - radius
    
    # Normalize direction
    dir_x = dist_vec_x / center_dist
    dir_y = dist_vec_y / center_dist
    
    # Stack manually
    direction = np.stack((dir_x, dir_y), axis=1)
    
    return direction, surface_dist

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
    """
    Python orchestration layer. Unpacks state, calls JIT functions, updates state.
    """
    pos = state['pos']
    vel = state['vel']
    diameters = state['diameters']
    goals = state['goals']
    
    num_agents = len(pos)
    if num_agents == 0: return state
    
    poi_locs = state.get('poi_locations')
    
    # 0. Goal Switching (Python Logic - Randomness is fast enough here)
    if poi_switch_chance > 0 and poi_locs is not None and len(poi_locs) > 0:
        switch_mask = np.random.rand(num_agents) < poi_switch_chance
        num_switching = np.sum(switch_mask)
        
        if num_switching > 0:
            random_indices = np.random.randint(0, len(poi_locs), size=int(num_switching))
            new_goals = poi_locs[random_indices]
            goals[switch_mask] = new_goals

    # 1. Map Parameters
    params = get_personality_params(rowdiness_val * 100)

    # 2. Variable Mass Calculation
    mass_array = (diameters / c.REF_DIAMETER)**2 * c.BASE_MASS

    # 3. Calculate Forces (CALLING JIT FUNCTIONS)
    F_goal = get_driving_force(pos, vel, goals, params['desired_speed'], params['tau'], mass_array)
    
    F_social = get_social_force(pos, params['radius'], params['social_strength'])
    
    F_contact = get_agent_contact_force(pos, diameters, c.AGENT_CONTACT_STIFFNESS)
    
    cell_w = c.LOGICAL_WIDTH / c.GRID_COLS
    cell_h = c.LOGICAL_HEIGHT / c.GRID_ROWS
    
    # Pass explicit constants to JIT function
    F_grid = get_grid_force(
        pos, grid_map, cell_w, cell_h, diameters, 
        c.WALL_REPULSION_STRENGTH, 
        c.WALL_STIFFNESS, 
        c.WALL_REPULSION_DIST,
        c.GRID_ROWS, c.GRID_COLS
    )
    
    F_noise = np.random.uniform(-1, 1, size=(num_agents, 2)) * params['noise']

    # 4. Integration
    F_total = F_goal + F_social + F_contact + F_grid + F_noise
    
    acceleration = F_total / mass_array[:, np.newaxis]
    vel += acceleration * c.DT
    
    max_speed = params['desired_speed'] * 1.5
    speed_mag = np.linalg.norm(vel, axis=1)
    limit_mask = speed_mag > max_speed
    if np.any(limit_mask):
        vel[limit_mask] = (vel[limit_mask] / speed_mag[limit_mask, np.newaxis]) * max_speed
        
    pos += vel * c.DT
    
    pos[:, 0] = np.clip(pos[:, 0], 0, c.LOGICAL_WIDTH - 0.1)
    pos[:, 1] = np.clip(pos[:, 1], 0, c.LOGICAL_HEIGHT - 0.1)

    # 5. Pressure Calculation
    crush_forces = F_social + F_contact
    crush_mag = np.linalg.norm(crush_forces, axis=1)

    c_idx = (pos[:, 0] / cell_w).astype(int)
    r_idx = (pos[:, 1] / cell_h).astype(int)
    c_idx = np.clip(c_idx, 0, c.GRID_COLS-1)
    r_idx = np.clip(r_idx, 0, c.GRID_ROWS-1)
    
    near_barrier_mask = np.zeros(num_agents, dtype=bool)
    # Simple check (non-JIT is fine for this mask calc, or we could JIT it)
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
