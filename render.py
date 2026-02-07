import streamlit as st
import numpy as np
import cv2
import constants as c # Importing the shared constants

# --- 1. UI GETTERS (Controller calls these) ---

def render_sidebar():
    """
    Draws the sidebar controls and returns the initial setup values.
    Returns: (num_agents, rowdiness)
    """
    st.sidebar.header("CrowdFlow Controls")
    
    # Setup Phase (Used for Init)
    # We use a unique key to prevent state collision if needed
    num_agents = st.sidebar.slider("Number of Agents", 100, 2000, 500)
    
    # Live Phase (Used for Tick)
    rowdiness = st.sidebar.slider("Rowdiness (Panic)", 0.0, 1.0, 0.0)
    
    return num_agents, rowdiness

def get_structure_grid():
    """
    Returns the 32x32 Integer Grid (0=Empty, 1=Wall).
    Currently hardcoded for testing, but eventually could come from a Map Editor.
    """
    grid = np.zeros((c.GRID_ROWS, c.GRID_COLS), dtype=int)
    
    # Hardcoded Test Map: "The Funnel"
    # Walls on edges
    grid[0, :] = c.ID_WALL
    grid[-1, :] = c.ID_WALL
    grid[:, 0] = c.ID_WALL
    grid[:, -1] = c.ID_WALL
    
    # Funnel shape logic
    mid = c.GRID_COLS // 2
    for r in range(10, 20):
        # Left diagonal
        grid[r, 0 : (mid - (r-8))] = c.ID_WALL
        # Right diagonal
        grid[r, (mid + (r-8)) : c.GRID_COLS] = c.ID_WALL
        
    return grid

# --- 2. RENDER LOGIC (SoA Style) ---

def get_color_from_pressure(pressure):
    """
    Helper to convert scalar pressure (0-255) into BGR color using constants.
    """
    p = max(0, min(255, int(pressure)))
    
    # Linear interpolation between SAFE -> WARN -> DANGER
    if p <= 128:
        # Green -> Orange
        fraction = p / 128.0
        # Interpolate between COLOR_AGENT_SAFE and COLOR_AGENT_WARN
        # Note: BGR tuples can be unpacked
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

def render_frame(positions, pressures, sizes, walls):
    """
    Main Rendering Function.
    Arguments:
      positions: Nx2 NumPy array (floats) [[x, y], ...]
      pressures: N NumPy array (floats/ints) [p1, p2, ...]
      sizes:     N NumPy array (floats) [s1, s2, ...] - Diameters
      walls:     32x32 NumPy array (Integers)
    
    Returns:
      640x640x3 NumPy array (RGB) ready for st.image
    """
    # 1. Init Canvas
    img = np.full((c.CANVAS_SIZE_PX, c.CANVAS_SIZE_PX, 3), c.COLOR_BG, dtype=np.uint8)
    
    # 2. Draw Walls (Vectorized Lookup)
    rows, cols = np.where(walls == c.ID_WALL)
    for r, c_idx in zip(rows, cols):
        x1 = c_idx * c.PIXELS_PER_CELL
        y1 = r * c.PIXELS_PER_CELL
        x2 = x1 + c.PIXELS_PER_CELL
        y2 = y1 + c.PIXELS_PER_CELL
        
        cv2.rectangle(img, (x1, y1), (x2, y2), c.COLOR_WALL, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), c.COLOR_GRID_LINE, 1)

    # 3. Draw Agents (Looping N times)
    scale_x = c.CANVAS_SIZE_PX / c.LOGICAL_WIDTH
    scale_y = c.CANVAS_SIZE_PX / c.LOGICAL_HEIGHT
    
    num_agents = len(positions)
    
    for i in range(num_agents):
        # A. Position
        x, y = positions[i]
        px = int(x * scale_x)
        py = int(y * scale_y)
        
        # B. Size (Radius = Diameter / 2)
        radius = int((sizes[i] * scale_x) / 2)
        radius = max(1, radius) # Safety
        
        # C. Color
        color = get_color_from_pressure(pressures[i])
        
        # D. Draw
        cv2.circle(img, (px, py), radius, color, -1)
        # Optional Outline
        if radius > 2:
            cv2.circle(img, (px, py), radius, (50, 50, 50), 1)

    # 4. Convert BGR (OpenCV) to RGB (Streamlit)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
