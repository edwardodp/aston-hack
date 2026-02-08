import streamlit as st
import numpy as np
import cv2
import os
import pandas as pd
import altair as alt  # <--- NEW IMPORT
from . import constants as c

# --- 1. UI GETTERS ---

def render_sidebar_controls(stop_callback, initial_rowdiness=0.0, initial_switch_chance=0.001):
    """
    Draws the sidebar controls for the LIVE simulation phase.
    """
    st.sidebar.markdown(
        "<h1 style='text-align: center;'>Live Controls</h1>", 
        unsafe_allow_html=True
    )
    
    # Live Physics Tweaks
    rowdiness = st.sidebar.slider("Rowdiness", 0.0, 1.0, float(initial_rowdiness), help="Simulate a more pushy and agitated crowd with a higher rowdiness.")

    switch_chance = st.sidebar.slider(
        "Wandering", 
        min_value=0.0, 
        max_value=0.005, 
        value=float(initial_switch_chance), 
        step=0.0001,
        format="%.4f",
        help="Probability per tick that an agent changes their destination."
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Safety Monitor") 
    chart_placeholder = st.sidebar.empty()
    st.sidebar.markdown("---")
    
    sb_col1, sb_col2, sb_col3 = st.sidebar.columns([1, 2, 1])
    
    with sb_col2:
        st.button(
            "Reset Simulation", 
            on_click=stop_callback, 
            type="secondary", 
            use_container_width=True
        )
    
    st.sidebar.info(
        """
        **Legend:**
        - 🟢 Green: Calm
        - 🟠 Orange: Uncomfortable
        - 🔴 Red: High Pressure
        """
    )
    
    return rowdiness, switch_chance, chart_placeholder

def render_chart(placeholder, pressure_history):
    """
    Updates the sidebar chart using Altair for fixed axes and labels.
    """
    if not pressure_history:
        return

    # Create a DataFrame with an explicit index for the X-axis
    df = pd.DataFrame({
        "Time": range(len(pressure_history)),
        "Pressure": pressure_history
    })
    
    # Create Altair Chart
    chart = alt.Chart(df).mark_line(color="#00e5ff").encode(
        # X-Axis Label
        x=alt.X('Time', axis=alt.Axis(title='Time (Frames)')),
        
        # Y-Axis Fixed Range (0-255) & Label
        y=alt.Y('Pressure', 
                scale=alt.Scale(domain=[0, 255]), 
                axis=alt.Axis(title='Avg Pressure'))
    ).properties(
        height=150
    )
    
    # Render with Streamlit
    placeholder.altair_chart(chart, use_container_width=True)

def get_structure_grid(csv_path=None):
    """
    Returns the 32x32 Integer Grid.
    Priority:
    1. Explicit 'csv_path' argument (if provided).
    2. The 'setup_map_select' from Streamlit Session State (if running).
    3. The Hardcoded Default (Fallback).
    """
    if csv_path and os.path.exists(csv_path):
        return load_grid_from_csv(csv_path)

    # --- 2. Context-Aware Lookup (THE FIX) ---
    # If no path is passed, check what the user selected in the dropdown.
    # We wrap this in a try-block to ensure safety if called outside Streamlit.
    try:
        # Check if we have a selection in session state
        selected_file = st.session_state.get("setup_map_select")
        
        # If the user selected a CSV file (not "Default"), try to load it
        if selected_file and selected_file.endswith(".csv"):
            # We assume the standard assets path here
            auto_path = os.path.join("assets/preset_maps", selected_file)
            if os.path.exists(auto_path):
                return load_grid_from_csv(auto_path)
    except Exception:
        # If session_state isn't accessible (e.g., running unit tests), ignore
        pass

    # --- 3. Hardcoded Default (Fallback) ---
    # This only runs if no path was passed AND no CSV is selected in the UI.
    grid = np.zeros((c.GRID_ROWS, c.GRID_COLS), dtype=int)
    
    # Borders
    grid[0, :] = c.ID_WALL
    grid[-1, :] = c.ID_WALL
    grid[:, 0] = c.ID_WALL
    grid[:, -1] = c.ID_WALL
    
    # grid[1:4, 11:21] = c.ID_POI
    #
    # # Default "Funnel" setup
    # grid[7, 12: 20] = c.ID_BARRIER
    
    return grid

def load_grid_from_csv(file_path):
    """
    Parses a .csv file into a numpy structure grid.
    
    Expected CSV Format:
    - Rows: 32 (c.GRID_ROWS)
    - Cols: 32 (c.GRID_COLS)
    - Values: Integers matching ID constants (0, 1, 2, 3)
    - Delimiter: Comma
    """
    try:
        # Load the CSV directly into a numpy integer array
        # delimiter="," specifies a standard CSV
        grid = np.loadtxt(file_path, delimiter=",", dtype=int)
        
        # Validate Dimensions
        expected_shape = (c.GRID_ROWS, c.GRID_COLS)
        if grid.shape != expected_shape:
            raise ValueError(f"Invalid Grid Size: Found {grid.shape}, expected {expected_shape}")
            
        return grid
        
    except Exception as e:
        # Fallback to a safe empty grid if loading fails
        print(f"Error loading map {file_path}: {e}")
        return np.zeros((c.GRID_ROWS, c.GRID_COLS), dtype=int)

def get_preview_image(grid):
    """
    Generates a static thumbnail image of the grid (Walls/POIs)
    for the UI preview.
    """
    # 1. Create a blank image (scaled up x10 for visibility)
    scale = 10
    h, w = grid.shape
    img = np.full((h * scale, w * scale, 3), c.COLOR_BG, dtype=np.uint8)

    # 2. Draw static elements
    for r in range(h):
        for col in range(w):
            val = grid[r, col]
            
            color = c.COLOR_BG
            if val == c.ID_WALL: color = c.COLOR_WALL
            elif val == c.ID_POI: color = c.COLOR_POI
            elif val == c.ID_BARRIER: color = c.COLOR_BARRIER
            
            # Draw if not empty
            if val != c.ID_NOTHING:
                x1, y1 = col * scale, r * scale
                x2, y2 = (col + 1) * scale, (r + 1) * scale
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    # 3. Convert BGR to RGB for Streamlit
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    # Determine Color based on ID
    color = c.COLOR_WALL # Default fallback
    
    if structure_id == c.ID_WALL:
        color = c.COLOR_WALL
    elif structure_id == c.ID_POI:
        color = c.COLOR_POI      # Dark Purple
    elif structure_id == c.ID_BARRIER:
        color = c.COLOR_BARRIER  # Dark Yellow

    # Draw Logic
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    # Optional: Draw grid lines for definition
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
