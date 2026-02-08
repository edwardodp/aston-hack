import streamlit as st
import numpy as np
import cv2
import os
import pandas as pd
import altair as alt
from . import constants as c

# --- 1. UI GETTERS ---
def render_sidebar_controls(stop_callback, initial_rowdiness=0.0, initial_switch_chance=0.001, music_active=False):
    st.sidebar.markdown(
        "<h1 style='text-align: center;'>Live Controls</h1>", 
        unsafe_allow_html=True
    )
    
    disable_msg = "Controlled by Audio Analysis"
    
    if music_active:
        st.sidebar.success("🎵 Music Integration Active")
        st.sidebar.caption("Crowd behavior is being driven by the music track.")
    
    rowdiness = st.sidebar.slider(
        "Rowdiness", 
        0.0, 1.0, 
        float(initial_rowdiness), 
        disabled=music_active,
        help=disable_msg if music_active else "Simulate a more pushy and agitated crowd."
    )

    switch_chance = st.sidebar.slider(
        "Wandering", 
        min_value=0.0, 
        max_value=0.005, 
        value=float(initial_switch_chance), 
        step=0.0001,
        format="%.4f",
        disabled=music_active,
        help=disable_msg if music_active else "Probability per tick that an agent changes their destination."
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
    if not pressure_history:
        return

    df = pd.DataFrame({
        "Time": range(len(pressure_history)),
        "Pressure": pressure_history
    })
    
    chart = alt.Chart(df).mark_line(color="#00e5ff").encode(
        x=alt.X('Time', axis=alt.Axis(title='Time (Frames)')),
        
        y=alt.Y('Pressure', 
                scale=alt.Scale(domain=[0, 255]), 
                axis=alt.Axis(title='Avg Pressure'))
    ).properties(
        height=150
    )
    
    placeholder.altair_chart(chart, use_container_width=True)

def get_structure_grid(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        return load_grid_from_csv(csv_path)

    try:
        selected_file = st.session_state.get("setup_map_select")
        
        if selected_file and selected_file.endswith(".csv"):
            auto_path = os.path.join("assets/preset_maps", selected_file)
            if os.path.exists(auto_path):
                return load_grid_from_csv(auto_path)
    except Exception:
        pass

    grid = np.zeros((c.GRID_ROWS, c.GRID_COLS), dtype=int)
    
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
    try:
        grid = np.loadtxt(file_path, delimiter=",", dtype=int)
        
        expected_shape = (c.GRID_ROWS, c.GRID_COLS)
        if grid.shape != expected_shape:
            raise ValueError(f"Invalid Grid Size: Found {grid.shape}, expected {expected_shape}")
            
        return grid
        
    except Exception as e:
        print(f"Error loading map {file_path}: {e}")
        return np.zeros((c.GRID_ROWS, c.GRID_COLS), dtype=int)

def get_preview_image(grid):
    scale = 10
    h, w = grid.shape
    img = np.full((h * scale, w * scale, 3), c.COLOR_BG, dtype=np.uint8)

    for r in range(h):
        for col in range(w):
            val = grid[r, col]
            
            color = c.COLOR_BG
            if val == c.ID_WALL: color = c.COLOR_WALL
            elif val == c.ID_POI: color = c.COLOR_POI
            elif val == c.ID_BARRIER: color = c.COLOR_BARRIER
            
            if val != c.ID_NOTHING:
                x1, y1 = col * scale, r * scale
                x2, y2 = (col + 1) * scale, (r + 1) * scale
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

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
    if structure_id == c.ID_NOTHING:
        return

    x1 = col * c.PIXELS_PER_CELL
    y1 = row * c.PIXELS_PER_CELL
    x2 = x1 + c.PIXELS_PER_CELL
    y2 = y1 + c.PIXELS_PER_CELL

    color = c.COLOR_WALL
    
    if structure_id == c.ID_WALL:
        color = c.COLOR_WALL
    elif structure_id == c.ID_POI:
        color = c.COLOR_POI
    elif structure_id == c.ID_BARRIER:
        color = c.COLOR_BARRIER

    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), c.COLOR_GRID_LINE, 1)


def render_agent(img, x, y, size, pressure):
    scale_x = c.CANVAS_SIZE_PX / c.LOGICAL_WIDTH
    scale_y = c.CANVAS_SIZE_PX / c.LOGICAL_HEIGHT

    px = int(x * scale_x)
    py = int(y * scale_y)

    radius = int((size * scale_x) / 2)
    radius = max(1, radius)

    color = get_color_from_pressure(pressure)

    cv2.circle(img, (px, py), radius, color, -1)
    
    if radius > 2:
        cv2.circle(img, (px, py), radius, (50, 50, 50), 1)

# --- 3. MAIN RENDER FUNCTION ---
def render_frame(positions, pressures, sizes, walls):
    img = np.full((c.CANVAS_SIZE_PX, c.CANVAS_SIZE_PX, 3), c.COLOR_BG, dtype=np.uint8)
    
    rows, cols = np.where(walls != c.ID_NOTHING)
    for r, col_idx in zip(rows, cols):
        structure_id = walls[r, col_idx]
        render_structure(img, r, col_idx, structure_id)

    num_agents = len(positions)
    for i in range(num_agents):
        x, y = positions[i]
        p = pressures[i]
        s = sizes[i]
        
        render_agent(img, x, y, s, p)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
