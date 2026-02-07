import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import math

dt = 0.1

st.title("Social Force Model Preview")

# 1. Setup Simulation Parameters
num_agents = st.sidebar.slider("Number of Agents", 2, 100, 9) # Default to 9 for a nice 3x3 grid
iterations = 1000

# --- SETUP: GEOMETRY (WALLS) ---
# Format: [x1, y1, x2, y2]
# We create a barrier at y=5 with a gap (door) between x=4 and x=6
walls = [
    [0, 5, 4, 5],   # Left Wall
    [6, 5, 10, 5],  # Right Wall
    # Optional: Corridor side walls (visual + physical)
    [0, 0, 0, 10],  # Far Left border
    [10, 0, 10, 10], # Far Right border
    [0, 0, 10, 0],
    [0, 10, 10,10]
]
# -------------------------------

# --- NEW: DYNAMIC GRID POSITIONING ---
# Calculate the side length of the grid (e.g., if 9 agents, we need a 3x3 grid)
grid_side = math.ceil(math.sqrt(num_agents))

# Generate evenly spaced points between 1 and 9 (keeping a margin from the 0-10 walls)
x_vals = np.linspace(1, 9, grid_side)
y_vals = np.linspace(0.5, 2.5, grid_side)

# Create the grid mesh
xx, yy = np.meshgrid(x_vals, y_vals)

# Flatten the grid and stack them into (x, y) pairs
grid_positions = np.vstack([xx.ravel(), yy.ravel()]).T

# Take only the first 'num_agents' positions (in case the grid is larger than needed)
positions = grid_positions[:num_agents]

# agent stats
velocities = np.zeros((num_agents, 2)) 

# 3. ACCELERATION (The result of forces)
# shape: (N, 2). Reset to zero every frame.
# acc = np.zeros((num_agents, 2)) 

# 4. GOALS (Where they want to go)
# shape: (N, 2). Let's set a single target for everyone (e.g., center of room)
goals = np.array([5.0, 9.0])
# -------------------------------------

def get_driving_force(pos, vel, goal, desired_speed=0.8, tau=0.5):
    """
    Calculates the force pulling agents toward the goal.
    
    pos: (N, 2) numpy array of current positions
    vel: (N, 2) numpy array of current velocities
    goal: (2,) numpy array of the target coordinate
    desired_speed: float, how fast they want to go (m/s)
    tau: float, relaxation time (reaction speed)
    """
    
    # 1. Vector pointing from Agent to Goal
    # Broadcasting: (2,) - (N, 2) works automatically in NumPy
    direction_vector = goal - pos 
    
    # 2. Distance to goal (Pythagoras theorem on the vectors)
    # axis=1 means we sum across coordinates (x^2 + y^2) for each row
    distance = np.linalg.norm(direction_vector, axis=1)
    
    # Avoid division by zero (if agent is exactly at the goal)
    # We add a tiny epsilon or set explicit logic. Here's a safe trick:
    distance[distance == 0] = 0.001 
    
    # 3. Normalize: Make the direction vector length 1 (Unit Vector)
    # We need to reshape distance to (N, 1) so we can divide the (N, 2) vectors
    unit_direction = direction_vector / distance[:, np.newaxis]
    
    # 4. Desired Velocity: Direction * Speed Limit
    desired_velocity = unit_direction * desired_speed
    
    # 5. The Force: (Target - Current) / Reaction Time
    # This is the "urge" to change speed/direction
    force = (desired_velocity - vel) / tau
    
    return force

def get_social_force(pos, interaction_radius=0.1, repulsion_strength=20.0):
    """
    Calculates the repulsive force between all pairs of agents.
    
    pos: (N, 2) numpy array of positions
    interaction_radius: The distance at which repulsion starts becoming significant
    repulsion_strength: The magnitude of the push
    """
    N = pos.shape[0]
    
    # 1. Calculate relative positions (N, N, 2)
    # diff[i, j] = pos[i] - pos[j]
    # This creates a 3D tensor representing the vector from every agent to every other agent
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    
    # 2. Calculate distances (N, N)
    dists = np.linalg.norm(diff, axis=2)
    
    # 3. Create the Identity Mask to ignore self-repulsion
    # We don't want agent i to repel itself. 
    # np.eye creates a matrix with 1s on the diagonal. We invert it to get 0s on diagonal.
    mask = 1 - np.eye(N)
    
    # 4. Calculate Repulsion Direction
    # Normalize the difference vectors. 
    # Add epsilon to dists to avoid division by zero for self-interaction (diagonal)
    directions = diff / (dists[:, :, np.newaxis] + 1e-8)
    
    # 5. Calculate Repulsion Magnitude
    # Simple exponential repulsion: Force = A * exp((radius - distance) / B)
    # Here we use a simplified inverse-distance model for clarity:
    # Force increases as distance decreases.
    
    # Only apply force if they are within interaction radius
    interaction_mask = dists < interaction_radius
    
    # Combine the "Not Self" mask and "Within Range" mask
    active_mask = mask * interaction_mask
    
    # Calculate magnitude: stronger when closer
    # We use exp(-distance) so it drops off smoothly
    magnitude = repulsion_strength * np.exp(-dists)
    
    # 6. Apply magnitude to direction
    # reshaping magnitude to (N, N, 1) to broadcast against (N, N, 2) directions
    force_vectors = directions * magnitude[:, :, np.newaxis] * active_mask[:, :, np.newaxis]
    
    # 7. Sum up all forces acting on each agent
    # Sum along axis 1 (the "other agents" axis) to get total force per agent
    total_force = np.sum(force_vectors, axis=1)
    
    return total_force

def get_wall_force(pos, walls, repulsion_strength=50.0, repulsion_range=0.3):
    """
    Calculates repulsion from a list of line segment walls.
    """
    total_force = np.zeros_like(pos)
    
    for wall in walls:
        start = np.array([wall[0], wall[1]])
        end = np.array([wall[2], wall[3]])
        
        # Vector representing the wall segment
        wall_vec = end - start
        wall_len_sq = np.dot(wall_vec, wall_vec)
        
        # Vector from wall start to agents
        # shape: (N, 2)
        agent_vec = pos - start
        
        # Project agent_vec onto wall_vec to find "t"
        # t is the relative position on the line (0=start, 1=end)
        # We verify shape compatibility for dot product
        t = np.sum(agent_vec * wall_vec, axis=1) / wall_len_sq
        
        # Clamp t to the segment [0, 1]
        # This ensures we find the closest point on the SEGMENT, not the infinite line
        t = np.clip(t, 0, 1)
        
        # Calculate the closest point on the segment for each agent
        # shape: (N, 2)
        closest_point = start + (t[:, np.newaxis] * wall_vec)
        
        # Vector from wall to agent
        dist_vec = pos - closest_point
        dist = np.linalg.norm(dist_vec, axis=1)
        
        # Normalize to get direction
        direction = dist_vec / (dist[:, np.newaxis] + 1e-8)
        
        # Calculate Repulsion (Exponential)
        # Only apply if close enough
        active_mask = dist < repulsion_range
        magnitude = repulsion_strength * np.exp(-dist)
        
        force = direction * magnitude[:, np.newaxis] * active_mask[:, np.newaxis]
        total_force += force
        
    return total_force

# 2. Create the placeholder for the simulation
frame_placeholder = st.empty()
stop_button = st.button("Stop Simulation")

# 3. The Simulation Loop
for i in range(iterations):
    if stop_button:
        break

    # --- SIMULATION LOGIC START ---
    
    # A. Calculate Driving Force
    # Let's say everyone wants to go to the center [5, 5]
    F_goal = get_driving_force(positions, velocities, goals)
    
    # B. Calculate Social Force (Repulsion)
    # Play with 'repulsion_strength' to make them more/less polite!
    F_social = get_social_force(positions, interaction_radius=0.5, repulsion_strength=20.0)

    # 3. Obstacle Force (Agent-Wall) -- NEW!
    F_wall = get_wall_force(positions, walls, repulsion_strength=100.0) # Strong walls!

    # C. Resultant Force (Vector Sum)
    F_total = F_goal + F_social + F_wall

    # B. Apply Physics (Euler Integration)
    # Acceleration = Force / Mass (assume mass = 1)
    acceleration = F_total / 3

    # Update Velocity
    velocities += acceleration * dt
    
    # Speed Limit (Optional but recommended for stability)
    speed = np.linalg.norm(velocities, axis=1)
    max_speed = 2.0
    mask = speed > max_speed
    velocities[mask] = (velocities[mask] / speed[mask, np.newaxis]) * max_speed
    

    # Update Position
    positions += velocities * dt

    # --- SIMULATION LOGIC END ---



    # 4. Rendering the frame
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # --- NEW: DRAW WALLS ---
    for wall in walls:
        # wall = [x1, y1, x2, y2]
        # Plot takes [x1, x2], [y1, y2]
        ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='black', linewidth=3)
    # -----------------------
    
    # Render the agents
    ax.scatter(positions[:, 0], positions[:, 1], s=100, color='dodgerblue', alpha=0.8, edgecolors='black')
    
    ax.scatter(goals[0], goals[1], s=100, color='lime', marker='x', label='Goal')
    
    # Setup the plot boundaries
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title(f"Step: {i}")
    ax.grid(True, linestyle='--', alpha=0.3) # Added grid for visual reference
    
    # Update the placeholder with the new plot
    frame_placeholder.pyplot(fig)
    plt.close(fig) # Important: Clean up memory

st.success("Simulation Complete")