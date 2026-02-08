import streamlit as st
from PIL import Image
from src import render
from src import loop
import os

# --- SETUP PAGE ---
favicon_image = Image.open("assets/crowd.png")
st.set_page_config(page_title="Crowd Flow", page_icon=favicon_image, layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "page" not in st.session_state:
    st.session_state.page = "setup" # Options: "setup", "simulation"

if "sim_params" not in st.session_state:
    st.session_state.sim_params = {} # Stores config from page 1

# --- SESSION STATE ---
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

# --- CALLBACKS ---
def start_simulation():
    # Save the slider value into session state so it persists
    st.session_state.sim_params["num_agents"] = st.session_state["setup_agents_slider"]
    st.session_state.sim_params["rowdiness_level"] = st.session_state["setup_rowdiness_slider"]
    st.session_state.page = "simulation"
    st.session_state.sim_running = True

def stop_simulation():
    st.session_state.sim_running = False
    st.session_state.page = "setup"
    # Clear physics state to force a reset on next start
    if "physics_state" in st.session_state:
        del st.session_state.physics_state
        
    if "structure_grid" in st.session_state.sim_params:
        del st.session_state.sim_params["structure_grid"]

main_container = st.empty()

# --- PAGE 1: SETUP ---
if st.session_state.page == "setup":
    with main_container.container():
        # 1. Centered Title
        st.markdown("<h1 style='text-align: center;'>Crowd Flow Simulation</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: gray;'>Simulate crowded events, save communities</h4>", unsafe_allow_html=True)
        st.markdown("---")

        # 2. Centered Inputs
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            with st.container(border=True):
                st.subheader("Initial Parameters")
                
                st.slider(
                    "Number of Agents", 
                    min_value=100, 
                    max_value=2000, 
                    value=500, 
                    key="setup_agents_slider"
                )
                
                st.write("") 
                
                st.slider(
                    "Crowd Rowdiness (Panic)", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.0, 
                    key="setup_rowdiness_slider"
                )
                
                st.write("") 
                
                st.subheader("Scene Selection")
                
                # Scan assets
                maps_dir = "assets/preset_maps"
                if not os.path.exists(maps_dir):
                    available_maps = []
                else:
                    available_maps = [f for f in os.listdir(maps_dir) if f.endswith(".csv")]

                if not available_maps:
                    st.warning("No .csv maps found! Using default grid.")
                    structure_grid = render.get_structure_grid(None)
                    st.session_state.sim_params["structure_grid"] = structure_grid
                else:
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        selected_file = st.selectbox("Choose a Layout Preset:", available_maps)
                        st.caption("Map Legend:")
                        st.markdown(
                            """
                            <div style="display: flex; gap: 10px; font-size: 0.8em;">
                                <div><span style="color: grey">■</span> Wall</div>
                                <div><span style="color: purple">■</span> Goal</div>
                                <div><span style="color: #FFD700">■</span> Barrier</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    # Load & Preview
                    selected_map_path = os.path.join(maps_dir, selected_file)
                    structure_grid = render.get_structure_grid(selected_map_path)
                    st.session_state.sim_params["structure_grid"] = structure_grid
                    
                    with c2:
                        if structure_grid is not None:
                            preview_img = render.get_preview_image(structure_grid)
                            st.image(preview_img, caption=f"Preview: {selected_file}", width=300)

                st.markdown("---")
                
                # Start Button
                sub_c1, sub_c2, sub_c3 = st.columns([1, 2, 1])
                with sub_c2:
                    st.button("Start Simulation", on_click=start_simulation, type="primary", use_container_width=True)

# --- PAGE 2: SIMULATION ---
elif st.session_state.page == "simulation":
    
    # 1. EXPLICITLY CLEAR THE SETUP PAGE
    # This removes the "ghost" elements before we start the blocking loop
    main_container.empty()
    
    # 2. Retrieve Config
    init_agents = st.session_state.sim_params.get("num_agents", 500)
    init_rowdiness = st.session_state.sim_params.get("rowdiness_level", 0.0)
    structure_grid = st.session_state.sim_params.get("structure_grid")

    if structure_grid is None:
        st.error("No structure grid found. Please go back to setup.")
        if st.button("Back to Setup"):
            stop_simulation()
            st.rerun()
    else:
        # 3. Sidebar
        rowdiness = render.render_sidebar_controls(
            stop_callback=stop_simulation, 
            initial_rowdiness=init_rowdiness
        )

        # 4. Simulation Layout
        st.markdown("<h2 style='text-align: center;'>Live Simulation</h2>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([2, 4, 2])
        with c2:
            canvas_placeholder = st.empty()
        
        # 5. Run the Loop
        loop.run_simulation(canvas_placeholder, init_agents, rowdiness, structure_grid)