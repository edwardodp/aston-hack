import streamlit as st
import time
from . import physics      # The Model
from . import render       # The View
from . import constants as c

def run_simulation(canvas_placeholder, num_agents, base_rowdiness, structure_grid, base_switch_chance, chart_placeholder):
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
    # CHECK FOR MUSIC
    music_data = st.session_state.sim_params.get("music_data")
    start_time = time.time() # Track real time for sync
    
    # --- 2. THE LOOP ---
    while True:
        if not st.session_state.get("sim_running", False):
            break
        
        # --- DYNAMIC PHYSICS PARAMS ---
        current_rowdiness = base_rowdiness
        current_switch_chance = base_switch_chance

        if music_data:
            elapsed = time.time() - start_time
            frame_idx = int(elapsed * music_data["fps"])
            
            if frame_idx < len(music_data["curve"]):
                # Rowdiness: Scale up music intensity
                music_intensity = music_data["curve"][frame_idx]
                current_rowdiness = music_intensity
                
                # 3. HYPE LOGIC: Kill wandering when energy is high
                # If music is > 40% intensity, we drastically reduce wandering.
                # If music is > 70% intensity, wandering is ZERO (Locked in).
                if music_intensity > 0.7:
                    current_switch_chance = 0.0
                elif music_intensity > 0.4:
                    current_switch_chance = base_switch_chance * 0.01 # 90% reduction
                else:
                    # Low energy: Standard wandering applies
                    current_switch_chance = base_switch_chance
                
            else:
                # Song over
                pass

        # B. SUB-TICKING
        for _ in range(c.SUB_TICKS):
            # Pass the current state, get the updated state back
            # Note: We also pass structure_grid because physics.tick needs it for wall forces
            state = physics.tick(state, current_rowdiness, structure_grid, current_switch_chance)
            
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
