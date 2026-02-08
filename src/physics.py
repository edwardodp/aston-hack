import numpy as np
from . import constants as c

# --- 1. CONFIGURATION HELPERS ---

def get_personality_params(rowdiness_percent):
    """
    Returns a dictionary of physics coefficients based on rowdiness (0-100).
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

# --- 2. PHYSICS FORCES (PURE FUNCTIONS) ---

def get_driving_force(pos, vel, goal, desired_speed, tau, mass_array):
    """
    Calculates self-driven force. 
    """
    direction_vector = goal - pos 
    distance = np.linalg.norm(direction_vector, axis=1)
    distance[distance == 0] = 0.001 
    
    unit_direction = direction_vector / distance[:, np.newaxis]
    desired_velocity = unit_direction * desired_speed
    
    # F = m * a
    force = (desired_velocity - vel) / tau
    return force * mass_array[:, np.newaxis]

def get_social_force(pos, interaction_radius, repulsion_strength):
    """
    Psychological Repulsion (Exponential).
    """
    N = pos.shape[0]
    if N == 0: return np.zeros((0, 2))

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    directions = diff / (dists[:, :, np.newaxis] + 1e-8)
    
    not_self_mask = 1 - np.eye(N)
    range_mask = dists < interaction_radius
    active_mask = not_self_mask * range_mask
    
    # Exponential decay
    magnitude = repulsion_strength * np.exp(-dists / (interaction_radius / 2.0))
    force_vectors = directions * magnitude[:, :, np.newaxis] * active_mask[:, :, np.newaxis]
    
    return np.sum(force_vectors, axis=1)

def get_agent_contact_force(pos, diameters, stiffness):
    """
    Physical Contact Force (Hooke's Law).
    Strictly handles body overlap.
    """
    N = pos.shape[0]
    if N == 0: return np.zeros((0, 2))
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Calculate sum of radii for every pair (N, N)
    radii = diameters / 2.0
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Check overlap: Distance < Radii Sum
    overlap = radii_sum - dists
    overlap_mask = overlap > 0
    
    # Boolean mask for identity to safely use bitwise AND
    not_self_mask = ~np.eye(N, dtype=bool)
    active_mask = overlap_mask & not_self_mask
    
    # Direction: Pushing apart
    directions = diff / (dists[:, :, np.newaxis] + 1e-8)
    
    # Force = stiffness * overlap
    magnitude = stiffness * overlap
    
    force_vectors = directions * magnitude[:, :, np.newaxis] * active_mask[:, :, np.newaxis]
    
    return np.sum(force_vectors, axis=1)

# --- 3. MODULAR GRID FORCE ---

def _get_wall_interaction(pos, diameters, cell_box):
    closest_x = np.maximum(cell_box[0], np.minimum(pos[:, 0], cell_box[2]))
    closest_y = np.maximum(cell_box[1], np.minimum(pos[:, 1], cell_box[3]))
    closest_points = np.stack([closest_x, closest_y], axis=1)

    dist_vec = pos - closest_points
    center_dist = np.linalg.norm(dist_vec, axis=1)

    degenerate_mask = center_dist < 0.001
    if np.any(degenerate_mask):
        dist_vec[degenerate_mask] = np.array([0.1, 0.0]) 
        center_dist[degenerate_mask] = 0.1

    radius = diameters / 2.0
    surface_dist = center_dist - radius
    direction = dist_vec / (center_dist[:, np.newaxis])
    
    return direction, surface_dist

def _calc_force_magnitude(surface_dist, repulsion_strength, repulsion_range):
    contact_force = np.zeros_like(surface_dist)
    overlap_mask = surface_dist < 0
    contact_force[overlap_mask] = -surface_dist[overlap_mask] * c.WALL_STIFFNESS

    eff_dist = np.maximum(0, surface_dist)
    soft_force = repulsion_strength * np.exp(-eff_dist / (repulsion_range / 2.0))
    
    return contact_force + soft_force

def get_grid_force(pos, grid_map, cell_w, cell_h, diameters, repulsion_strength):
    N = pos.shape[0]
    total_force = np.zeros_like(pos)
    if grid_map is None: return total_force

    col_indices = (pos[:, 0] / cell_w).astype(int)
    row_indices = (pos[:, 1] / cell_h).astype(int)
    
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            r_neigh = np.clip(row_indices + dy, 0, c.GRID_ROWS - 1)
            c_neigh = np.clip(col_indices + dx, 0, c.GRID_COLS - 1)
            
            # --- CHANGE 1: Treat POI as a Wall for collision purposes ---
            # If it is a WALL, BARRIER, or POI (Stage), you cannot walk through it.
            is_wall = (grid_map[r_neigh, c_neigh] == c.ID_WALL) | \
                      (grid_map[r_neigh, c_neigh] == c.ID_BARRIER) | \
                      (grid_map[r_neigh, c_neigh] == c.ID_POI)
            
            if not np.any(is_wall): continue
                
            cell_x1 = c_neigh * cell_w
            cell_y1 = r_neigh * cell_h
            
            direction, surf_dist = _get_wall_interaction(
                pos, diameters, 
                [cell_x1, cell_y1, cell_x1 + cell_w, cell_y1 + cell_h]
            )
            
            magnitude = _calc_force_magnitude(surf_dist, repulsion_strength, c.WALL_REPULSION_DIST)
            
            active_mask = is_wall & (surf_dist < c.WALL_REPULSION_DIST)
            
            force_contribution = direction * magnitude[:, np.newaxis] * active_mask[:, np.newaxis]
            total_force += force_contribution
            
    return total_force

# --- 4. STATE MANAGEMENT ---

def init_state(num_agents, grid_map):
    cell_w = c.LOGICAL_WIDTH / c.GRID_COLS
    cell_h = c.LOGICAL_HEIGHT / c.GRID_ROWS
    
    # 1. Find all potential target points (POI Cells)
    poi_locations = []
    for r in range(c.GRID_ROWS):
        for col in range(c.GRID_COLS):
            if grid_map[r, col] == c.ID_POI:
                # Target is the center of the POI cell
                cx = (col * cell_w) + (cell_w / 2)
                cy = (r * cell_h) + (cell_h / 2)
                poi_locations.append([cx, cy])
    
    # Pre-calculate POI array for storage and assignment
    poi_array = np.array(poi_locations) if poi_locations else np.empty((0, 2))
    
    # 2. Assign Goals
    if len(poi_array) > 0:
        # Randomly assign each agent to one of the POI locations
        # This creates sub-crowds heading to different destinations
        random_indices = np.random.randint(0, len(poi_array), size=num_agents)
        goals = poi_array[random_indices] # Shape: (num_agents, 2)
    else:
        # Fallback: Everyone goes to the center
        center = np.array([c.LOGICAL_WIDTH / 2.0, c.LOGICAL_HEIGHT / 2.0])
        # Broadcast center to shape (num_agents, 2)
        goals = np.full((num_agents, 2), center)

    # 3. Spawn Agents (Standard Logic)
    valid_positions = []
    attempts = 0
    max_attempts = num_agents * 50
    
    # Agents spawn in Empty space (0).
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
        'poi_locations': poi_array # Store for runtime swapping
    }

def tick(state, rowdiness_val, grid_map, poi_switch_chance=0.01):
    pos = state['pos']
    vel = state['vel']
    diameters = state['diameters']
    goals = state['goals']
    
    num_agents = len(pos)
    if num_agents == 0: return state
    
    poi_locs = state.get('poi_locations')
    
    # Only switch if we have POIs and the chance is non-zero
    if poi_switch_chance > 0 and poi_locs is not None and len(poi_locs) > 0:
        # Generate a boolean mask for agents that should switch this tick
        switch_mask = np.random.rand(num_agents) < poi_switch_chance
        num_switching = np.sum(switch_mask)
        
        if num_switching > 0:
            # Pick new random goals for the switching agents
            random_indices = np.random.randint(0, len(poi_locs), size=num_switching)
            new_goals = poi_locs[random_indices]
            
            # Update the goals in the state array
            goals[switch_mask] = new_goals

    # 1. Map Parameters
    params = get_personality_params(rowdiness_val * 100)

    # 2. Variable Mass Calculation
    mass_array = (diameters / c.REF_DIAMETER)**2 * c.BASE_MASS

    # 3. Calculate Forces
    F_goal = get_driving_force(pos, vel, goals, params['desired_speed'], params['tau'], mass_array)
    F_social = get_social_force(pos, params['radius'], params['social_strength'])
    F_contact = get_agent_contact_force(pos, diameters, c.AGENT_CONTACT_STIFFNESS)
    
    cell_w = c.LOGICAL_WIDTH / c.GRID_COLS
    cell_h = c.LOGICAL_HEIGHT / c.GRID_ROWS
    F_grid = get_grid_force(pos, grid_map, cell_w, cell_h, diameters, c.WALL_REPULSION_STRENGTH)
    
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
    # We only count forces from OTHER AGENTS (Social + Contact).
    crush_forces = F_social + F_contact
    crush_mag = np.linalg.norm(crush_forces, axis=1)

    # Detect if agents are near a BARRIER (Safety Rail).
    c_idx = (pos[:, 0] / cell_w).astype(int)
    r_idx = (pos[:, 1] / cell_h).astype(int)
    
    c_idx = np.clip(c_idx, 0, c.GRID_COLS-1)
    r_idx = np.clip(r_idx, 0, c.GRID_ROWS-1)
    
    near_barrier_mask = np.zeros(num_agents, dtype=bool)
    
    # Simple 3x3 check for nearby barriers
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny = np.clip(r_idx + dy, 0, c.GRID_ROWS-1)
            nx = np.clip(c_idx + dx, 0, c.GRID_COLS-1)
            near_barrier_mask |= (grid_map[ny, nx] == c.ID_BARRIER)

    # 6. Apply Relief & Sensitivity
    relief_factor = np.ones(num_agents)
    # Agents near barriers feel 80% less pressure (Safety Rail effect)
    relief_factor[near_barrier_mask] = 0.2 
    
    # FIX 1: Apply the relief_factor
    # FIX 2: Lower sensitivity from 0.25 to 0.10 (Requires 2500N to hit max red)
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
