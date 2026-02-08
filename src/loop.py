import streamlit as st
import time
from . import physics
from . import render
from . import constants as c

def run_simulation(canvas_placeholder, num_agents, base_rowdiness, structure_grid, base_switch_chance, chart_placeholder):
    # --- 1. INITIALIZATION CHECK ---
    if "physics_state" not in st.session_state:
        initial_state = physics.init_state(num_agents, structure_grid)
        st.session_state.physics_state = initial_state
        st.session_state.physics_initialized = True
    
    state = st.session_state.physics_state
    
    pressure_history = []
    frame_count = 0
    music_data = st.session_state.sim_params.get("music_data")
    start_time = time.time()
    
    # --- 2. THE LOOP ---
    while True:
        if not st.session_state.get("sim_running", False):
            break
        
        current_rowdiness = base_rowdiness
        current_switch_chance = base_switch_chance

        if music_data:
            elapsed = time.time() - start_time
            frame_idx = int(elapsed * music_data["fps"])
            
            if frame_idx < len(music_data["curve"]):
                music_intensity = music_data["curve"][frame_idx]
                current_rowdiness = music_intensity
                
                if music_intensity > 0.7:
                    current_switch_chance = 0.0
                elif music_intensity > 0.4:
                    current_switch_chance = base_switch_chance * 0.01
                else:
                    current_switch_chance = base_switch_chance
                
            else:
                pass

        for _ in range(c.SUB_TICKS):
            state = physics.tick(state, current_rowdiness, structure_grid, current_switch_chance)
            
        pos_array  = physics.get_agents_pos(state)
        pres_array = physics.get_agents_pressure(state)
        size_array = physics.get_agents_size(state)
        wall_grid  = physics.get_walls(state)
        
        current_avg_pressure = physics.get_average_pressure(state)
        pressure_history.append(current_avg_pressure)
        
        if len(pressure_history) > 200:
            pressure_history.pop(0)

        frame = render.render_frame(pos_array, pres_array, size_array, wall_grid)
        canvas_placeholder.image(frame, channels="RGB")
        
        if frame_count % 5 == 0:
            render.render_chart(chart_placeholder, pressure_history)
            
        frame_count += 1
        
        time.sleep(c.DT)

    st.session_state.physics_state = state
