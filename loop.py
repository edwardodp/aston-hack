import streamlit as st
import time
import physics      # The Model
import render       # The View
import constants as c

def run_simulation(canvas_placeholder, num_agents, rowdiness, structure_grid):
    """
    The main Game Loop. 
    This function BLOCKS execution (contains while True).
    """
    
    # --- 1. INITIALIZATION CHECK ---
    # We use session_state to ensure we don't re-init logic 
    # if the script reruns just because a slider moved.
    if "physics_initialized" not in st.session_state:
        physics.init_simulation(num_agents, structure_grid)
        st.session_state.physics_initialized = True
    
    # --- 2. THE LOOP ---
    while True:
        # A. Check Stop Condition
        # If the user clicked "Stop" in app.py, the session state changes.
        # However, inside a blocking loop, we rely on the script being killed/rerun 
        # by Streamlit interactions. 
        # For safety, we can check the state if app.py set it.
        if not st.session_state.get("sim_running", False):
            break

        # B. SUB-TICKING (Physics runs faster than Rendering)
        # 5 Physics Ticks per 1 Render Frame
        for _ in range(1):
            physics.tick(rowdiness)
            
        # C. GET DATA (Pull from Model)
        # SoA (Structure of Arrays) style
        pos_array = physics.get_agents_pos()
        pres_array = physics.get_agents_pressure()
        size_array = physics.get_agents_size()
        wall_grid = physics.get_walls()
        
        # D. RENDER (Push to View)
        # We pass the raw arrays directly to the renderer
        frame = render.render_frame(pos_array, pres_array, size_array, wall_grid)
        
        # E. DISPLAY (Update Streamlit)
        canvas_placeholder.image(frame, channels="RGB")
        
        # F. YIELD (Cap at ~60 FPS)
        time.sleep(c.DT)
