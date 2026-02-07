import streamlit as st
import render
import loop

# --- SETUP PAGE ---
st.set_page_config(page_title="CrowdFlow", layout="wide")
st.title("CrowdFlow Simulation")

# --- SESSION STATE ---
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

def start_cb():
    st.session_state.sim_running = True

def stop_cb():
    st.session_state.sim_running = False
    # Optional: Reset physics init flag so next start is fresh
    if "physics_initialized" in st.session_state:
        del st.session_state.physics_initialized

# --- SIDEBAR (Render Module) ---
# We get the params, but we only use them if we are starting/running
num_agents, rowdiness = render.render_sidebar()

# --- MAIN CONTROLS ---
col1, col2 = st.columns([1, 5])
with col1:
    if not st.session_state.sim_running:
        st.button("Start Simulation", on_click=start_cb, type="primary")
    else:
        st.button("Stop Simulation", on_click=stop_cb, type="secondary")

# --- CANVAS AREA ---
canvas_placeholder = st.empty()

# --- EXECUTION ---
if st.session_state.sim_running:
    # 1. Get the Map (Snapshot at start)
    # We get this from the render module (which might have a map editor later)
    structure_grid = render.get_structure_grid()
    
    # 2. Handover to the Loop Module
    # This will BLOCK here until the simulation is stopped
    loop.run_simulation(canvas_placeholder, num_agents, rowdiness, structure_grid)
