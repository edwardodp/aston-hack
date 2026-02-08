import streamlit as st
from PIL import Image
from src import render
from src import loop
from src import optimiser  # Imported the optimization engine
import os

# --- SETUP PAGE CONFIG ---
try:
    favicon_image = Image.open("assets/crowd.png")
    st.set_page_config(page_title="Crowd Flow", page_icon=favicon_image, layout="wide")
except:
    st.set_page_config(page_title="Crowd Flow", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "page" not in st.session_state:
    st.session_state.page = "setup"  # Options: "setup", "simulation"

if "sim_params" not in st.session_state:
    st.session_state.sim_params = {}  # Stores config to pass to simulation

if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

# Flag to track if we are currently using an optimized map
if "is_optimized" not in st.session_state:
    st.session_state.is_optimized = False

# --- CALLBACKS ---
def start_simulation():
    # 1. Save Slider Values 
    st.session_state.sim_params["num_agents"] = st.session_state["setup_agents_slider"]
    st.session_state.sim_params["rowdiness_level"] = st.session_state["setup_rowdiness_slider"]
    st.session_state.sim_params["switch_chance"] = st.session_state["setup_switch_slider"]
    
    # 2. Map Selection Logic
    # If the map is NOT optimized, we load it from the file selector.
    # If it IS optimized, the grid is already stored in sim_params["structure_grid"], so we skip loading.
    if not st.session_state.is_optimized:
        maps_dir = "assets/preset_maps"
        selected_file = st.session_state.get("setup_map_select")
        
        if selected_file and os.path.exists(os.path.join(maps_dir, selected_file)):
            path = os.path.join(maps_dir, selected_file)
            grid = render.get_structure_grid(path)
        else:
            grid = render.get_structure_grid(None)
            
        st.session_state.sim_params["structure_grid"] = grid

    # 3. Switch Page
    st.session_state.page = "simulation"
    st.session_state.sim_running = True
    # No explicit rerun needed here as button click triggers update

def stop_simulation():
    st.session_state.sim_running = False
    st.session_state.page = "setup"
    
    # Clear physics state to force a complete reset of positions
    if "physics_state" in st.session_state:
        del st.session_state.physics_state
    
    # NOTE: We do NOT delete "structure_grid" if it is optimized, 
    # so the user preserves their AI layout when returning to setup.
    if not st.session_state.is_optimized and "structure_grid" in st.session_state.sim_params:
        del st.session_state.sim_params["structure_grid"]

def reset_optimization_cb():
    """Clear the optimized map and revert to original"""
    st.session_state.is_optimized = False
    # Remove the stored grid so the preview falls back to the CSV selection
    if "structure_grid" in st.session_state.sim_params:
        del st.session_state.sim_params["structure_grid"]

# --- MAIN CONTAINER ---
main_interface = st.empty()

# --- PAGE 1: SETUP ---
if st.session_state.page == "setup":
    
    with main_interface.container():
        
        # 1. Title Area
        logo_c1, logo_c2, logo_c3 = st.columns([1, 3, 1])
        with logo_c2:
            try:
                st.image("assets/titleAndIcon.png", use_container_width=True)
            except:
                st.title("Crowd Flow")
        
        st.markdown("<h4 style='text-align: center; color: gray;'>Simulate crowded events, save communities</h4>", unsafe_allow_html=True)
        st.markdown("---")

        # 2. Controls Area
        col1, col2, col3 = st.columns([1, 1, 1])

        # --- LEFT COLUMN: METRICS ---
        with col1:
            st.markdown("### 📡 System Status")
            st.metric(label="Physics Engine", value="Verlet", delta="Active")
            st.metric(label="Optimization Engine", value="Monte Carlo", delta="Ready")
            st.markdown("---")
            st.markdown("**Capabilities:**")
            st.markdown("- *Collision Prediction*")
            st.markdown("- *Social Force Model*")
            st.markdown("- *Automated Barrier Placement*")
            st.caption("Build v1.0.0 | State: Ready")
        
        # --- MIDDLE COLUMN: INPUTS (Fragmented) ---
        with col2:
            with st.container(border=True):
                
                # ISOLATED FRAGMENT: Interactions here won't reload the whole page
                @st.fragment
                def render_setup_interface():
                    st.markdown("<h3 style='text-align: center;'>Configuration</h3>", unsafe_allow_html=True)
                    
                    # A. Sliders
                    st.slider("Number of Agents", 100, 2000, 500, key="setup_agents_slider")
                    st.write("") 
                    st.slider(
                        "Crowd Rowdiness (Panic)", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.0, 
                        key="setup_rowdiness_slider"
                    )
                    st.write("")
                    st.slider(
                        "Wandering (Goal Switch)", 
                        min_value=0.0, 
                        max_value=0.005, 
                        value=0.001, 
                        step=0.0001,
                        format="%.4f",
                        key="setup_switch_slider"
                    )
                    st.write("")
                    
                    # B. Map Selection
                    st.subheader("Venue Selection")
                    
                    maps_dir = "assets/preset_maps"
                    available_maps = []
                    if os.path.exists(maps_dir):
                        available_maps = [f for f in os.listdir(maps_dir) if f.endswith(".csv")]
                    
                    # Store selection in session state key
                    selected_file = st.selectbox(
                        "Choose Layout:", 
                        available_maps if available_maps else ["Default"],
                        key="setup_map_select"
                    )

                    # Determine Current Grid for Preview
                    # 1. If Optimized: Use the stored result
                    if st.session_state.is_optimized:
                        current_grid = st.session_state.sim_params.get("structure_grid")
                        st.success("✅ Map Optimized by AI")
                    
                    # 2. If Not Optimized: Load from File
                    else:
                        if available_maps and selected_file in available_maps:
                            path = os.path.join(maps_dir, selected_file)
                            current_grid = render.get_structure_grid(path)
                        else:
                            current_grid = render.get_structure_grid(None)

                    # C. Map Preview Image
                    if current_grid is not None:
                        preview_img = render.get_preview_image(current_grid)
                        st.image(preview_img, caption="Venue Preview", use_container_width=True)

                    # D. Optimization Controls
                    st.write("")
                    opt_col1, opt_col2 = st.columns([1, 1])
                    
                    with opt_col1:
                        # Only show Optimize button if not already optimized
                        if not st.session_state.is_optimized:
                            if st.button("✨ Auto-Optimise", help="Run AI to find best barrier placement"):
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                viz_placeholder = st.empty()
                                
                                # Run Optimizer
                                best_grid = optimiser.run_optimisation(
                                    status_text,
                                    viz_placeholder,
                                    st.session_state["setup_agents_slider"],
                                    max_iter=50,  # Adjusted for speed
                                    patience=10,
                                    default_grid=current_grid
                                )
                                
                                # Store Result & Set Flag
                                st.session_state.sim_params["structure_grid"] = best_grid
                                st.session_state.is_optimized = True
                                st.rerun() # Rerun fragment to update preview
                    
                    with opt_col2:
                        # Show Reset button if optimized
                        if st.session_state.is_optimized:
                            if st.button("Reset Map"):
                                reset_optimization_cb()
                                st.rerun()

                # Render the Fragment
                render_setup_interface()

                st.markdown("---")
                
                # E. Start Button (Global Scope)
                st.button("🚀 Start Simulation", on_click=start_simulation, type="primary", use_container_width=True)

        # --- RIGHT COLUMN: GUIDE ---
        with col3:
            st.markdown("### 📋 User Guide")
            with st.expander("How to use", expanded=True):
                st.markdown("""
                1. **Setup**: Choose agent count and panic level.
                2. **Venue**: Select a map from the dropdown.
                3. **Optimise**: Click 'Auto-Optimise' to let the AI place barriers.
                4. **Run**: Click Start to watch the physics live.
                """)
            st.info("💡 **Tip:** Optimization runs a fast 'headless' simulation to test hundreds of layouts.")

# --- PAGE 2: SIMULATION ---
elif st.session_state.page == "simulation":
    
    # 1. Forcefully clear the Setup UI
    main_interface.empty()
    
    # 2. Retrieve Config
    init_agents = st.session_state.sim_params.get("num_agents", 500)
    init_rowdiness = st.session_state.sim_params.get("rowdiness_level", 0.0)
    init_switch = st.session_state.sim_params.get("switch_chance", 0.001)
    
    structure_grid = st.session_state.sim_params.get("structure_grid")

    if structure_grid is None:
        st.error("No map data found. Please return to setup.")
        if st.button("Back to Setup"):
            stop_simulation()
            st.rerun()
    else:
        # 3. Sidebar Controls
        rowdiness, switch_chance = render.render_sidebar_controls(
            stop_callback=stop_simulation, 
            initial_rowdiness=init_rowdiness,
            initial_switch_chance=init_switch
        )

        # 4. Simulation Layout
        st.markdown("<h2 style='text-align: center;'>Live Simulation</h2>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            canvas_placeholder = st.empty()
        
        # 5. Run the Loop
        loop.run_simulation(canvas_placeholder, init_agents, rowdiness, structure_grid, switch_chance)