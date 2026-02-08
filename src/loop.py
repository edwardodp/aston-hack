import streamlit as st
import time
from . import physics      # The Model
from . import render       # The View
from . import constants as c

def run_simulation(canvas_placeholder, num_agents, rowdiness, structure_grid, switch_chance):
    """
    The main Game Loop. 
    This function BLOCKS execution (contains while True).
    """
    
    # --- 1. INITIALIZATION CHECK ---
    # We use session_state to persist the simulation state (positions, velocities)
    # even if Streamlit reruns the script (e.g., when you move a slider).
    if "physics_state" not in st.session_state:
        # Create the initial state dictionary
        initial_state = physics.init_state(num_agents, structure_grid)
        st.session_state.physics_state = initial_state
        st.session_state.physics_initialized = True
    
    # Retrieve the state object to use in the loop
    state = st.session_state.physics_state
    
    # --- 2. THE LOOP ---
    while True:
        # A. Check Stop Condition
        # If the user clicked "Stop" in app.py, the session state changes.
        if not st.session_state.get("sim_running", False):
            break

        # B. SUB-TICKING (Physics runs faster than Rendering)
        # We run multiple small physics steps for every 1 frame drawn to screen.
        for _ in range(c.SUB_TICKS):
            # Pass the current state, get the updated state back
            # Note: We also pass structure_grid because physics.tick needs it for wall forces
            state = physics.tick(state, rowdiness, structure_grid, switch_chance)
            
        # C. GET DATA (Pull from Model)
        # The getters now require the 'state' dictionary to know what to read
        pos_array  = physics.get_agents_pos(state)
        pres_array = physics.get_agents_pressure(state)
        size_array = physics.get_agents_size(state)
        wall_grid  = physics.get_walls(state)
        
        # D. RENDER (Push to View)
        # We pass the raw arrays directly to the renderer
        frame = render.render_frame(pos_array, pres_array, size_array, wall_grid)
        
        # E. DISPLAY (Update Streamlit)
        canvas_placeholder.image(frame, channels="RGB")
        
        # F. YIELD (Cap at ~60 FPS)
        time.sleep(c.DT)

    # Optional: Save the final state back to session (if you want to pause/resume later)
    st.session_state.physics_state = state
