import streamlit as st
from src import render
from src import loop

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
    
    if "physics_state" in st.session_state:
        del st.session_state.physics_state
        
    if "physics_initialized" in st.session_state:
        del st.session_state.physics_initialized

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
    structure_grid = render.get_structure_grid()
    
    loop.run_simulation(canvas_placeholder, num_agents, rowdiness, structure_grid)
