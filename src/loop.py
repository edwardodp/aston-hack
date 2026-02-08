import streamlit as st
import time
from . import physics      # The Model
from . import render       # The View
from . import constants as c

# FIX: Added 'chart_placeholder' argument
def run_simulation(canvas_placeholder, num_agents, rowdiness, structure_grid, switch_chance, chart_placeholder):
    """
    The main Game Loop. 
    """
    
    # --- 1. INITIALIZATION CHECK ---
    if "physics_state" not in st.session_state:
        initial_state = physics.init_state(num_agents, structure_grid)
        st.session_state.physics_state = initial_state
        st.session_state.physics_initialized = True
    
    state = st.session_state.physics_state
    
    # --- NEW: Metric History ---
    pressure_history = []
    frame_count = 0
    
    # --- 2. THE LOOP ---
    while True:
        if not st.session_state.get("sim_running", False):
            break

        # B. SUB-TICKING
        for _ in range(c.SUB_TICKS):
            state = physics.tick(state, rowdiness, structure_grid, switch_chance)
            
        # C. GET DATA
        pos_array  = physics.get_agents_pos(state)
        pres_array = physics.get_agents_pressure(state)
        size_array = physics.get_agents_size(state)
        wall_grid  = physics.get_walls(state)
        
        # --- NEW: Capture Metric ---
        # Get the single source of truth metric
        current_avg_pressure = physics.get_average_pressure(state)
        pressure_history.append(current_avg_pressure)
        
        # Keep history manageable (last 200 points)
        if len(pressure_history) > 200:
            pressure_history.pop(0)

        # D. RENDER CANVAS (Every Frame)
        frame = render.render_frame(pos_array, pres_array, size_array, wall_grid)
        canvas_placeholder.image(frame, channels="RGB")
        
        # E. RENDER CHART (Every 5 Frames)
        # Updating charts is slow, so we throttle it to maintain FPS
        if frame_count % 5 == 0:
            render.render_chart(chart_placeholder, pressure_history)
            
        frame_count += 1
        
        # F. YIELD
        time.sleep(c.DT)

    st.session_state.physics_state = state
