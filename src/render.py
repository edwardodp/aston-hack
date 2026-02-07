import streamlit as st
import numpy as np
import cv2
import constants as c

# --- 1. UI GETTERS ---

def render_sidebar():
    """
    Draws the sidebar controls and returns the initial setup values.
    Returns: (num_agents, rowdiness)
    """
    st.sidebar.header("CrowdFlow Controls")
    
    # Setup Phase
    num_agents = st.sidebar.slider("Number of Agents", 100, 2000, 500)
    
    # Live Phase
    rowdiness = st.sidebar.slider("Rowdiness (Panic)", 0.0, 1.0, 0.0)
    
    return num_agents, rowdiness

def get_structure_grid():
    """
    Returns the 32x32 Integer Grid (0=Empty, 1=Wall).
    Hardcoded 'Funnel' map for testing.
    """
    grid = np.zeros((c.GRID_ROWS, c.GRID_COLS), dtype=int)
    
    # Borders
    grid[0, :] = c.ID_WALL
    grid[-1, :] = c.ID_WALL
    grid[:, 0] = c.ID_WALL
    grid[:, -1] = c.ID_WALL
    
    # Funnel
    mid = c.GRID_COLS // 2
    for r in range(10, 20):
        grid[r, 0 : (mid - (r-8))] = c.ID_WALL
        grid[r, (mid + (r-8)) : c.GRID_COLS] = c.ID_WALL
        
    return grid

# --- 2. RENDER HELPERS ---

def get_color_from_pressure(pressure):
    """
    Helper to convert scalar pressure (0-255) into BGR color using constants.
    """
    p = max(0, min(255, int(pressure)))
    
    if p <= 128:
        # Green -> Orange
        fraction = p / 128.0
        b = int(c.COLOR_AGENT_SAFE[0] * (1-fraction) + c.COLOR_AGENT_WARN[0] * fraction)
        g = int(c.COLOR_AGENT_SAFE[1] * (1-fraction) + c.COLOR_AGENT_WARN[1] * fraction)
        r = int(c.COLOR_AGENT_SAFE[2] * (1-fraction) + c.COLOR_AGENT_WARN[2] * fraction)
    else:
        # Orange -> Red
        fraction = (p - 128) / 128.0
        b = int(c.COLOR_AGENT_WARN[0] * (1-fraction) + c.COLOR_AGENT_DANGER[0] * fraction)
        g = int(c.COLOR_AGENT_WARN[1] * (1-fraction) + c.COLOR_AGENT_DANGER[1] * fraction)
        r = int(c.COLOR_AGENT_WARN[2] * (1-fraction) + c.COLOR_AGENT_DANGER[2] * fraction)
        
    return (b, g, r)

def render_structure(img, row, col, structure_id):
    """
    Draws a single grid cell structure onto the image.
    """
    if structure_id == c.ID_NOTHING:
        return

    # Calculate Pixel Coordinates
    x1 = col * c.PIXELS_PER_CELL
    y1 = row * c.PIXELS_PER_CELL
    x2 = x1 + c.PIXELS_PER_CELL
    y2 = y1 + c.PIXELS_PER_CELL

    # Determine Color/Shape based on ID
    color = c.COLOR_WALL # Default
    
    if structure_id == c.ID_WALL:
        color = c.COLOR_WALL
        # Future: If ID_EXIT, color = Green, etc.

    # Draw Logic
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), c.COLOR_GRID_LINE, 1)


def render_agent(img, x, y, size, pressure):
    """
    Draws a single agent onto the image.
    """
    # Scaling Factors
    scale_x = c.CANVAS_SIZE_PX / c.LOGICAL_WIDTH
    scale_y = c.CANVAS_SIZE_PX / c.LOGICAL_HEIGHT

    # Coordinate Transform
    px = int(x * scale_x)
    py = int(y * scale_y)

    # Size Transform (Diameter -> Radius)
    radius = int((size * scale_x) / 2)
    radius = max(1, radius)

    # Color Logic
    color = get_color_from_pressure(pressure)

    # Draw
    cv2.circle(img, (px, py), radius, color, -1)
    
    # Optional Outline (for better visibility)
    if radius > 2:
        cv2.circle(img, (px, py), radius, (50, 50, 50), 1)

# --- 3. MAIN RENDER FUNCTION ---

def render_frame(positions, pressures, sizes, walls):
    """
    Main Rendering Function.
    Iterates through SoA data and calls helpers.
    """
    # 1. Init Canvas
    img = np.full((c.CANVAS_SIZE_PX, c.CANVAS_SIZE_PX, 3), c.COLOR_BG, dtype=np.uint8)
    
    # 2. Draw Structures
    # Optimization: Only iterate over non-empty cells
    rows, cols = np.where(walls != c.ID_NOTHING)
    for r, col_idx in zip(rows, cols):
        structure_id = walls[r, col_idx]
        render_structure(img, r, col_idx, structure_id)

    # 3. Draw Agents
    num_agents = len(positions)
    for i in range(num_agents):
        # Extract individual values from SoA
        x, y = positions[i]
        p = pressures[i]
        s = sizes[i]
        
        render_agent(img, x, y, s, p)

    # 4. Convert to RGB for Streamlit
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
