import streamlit as st
from PIL import Image
import os

# Internal Modules
from src import render
from src import loop
from src import optimiser

# --- SETUP PAGE CONFIG ---
# Use a try-except for the image in case the file is missing during dev
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
    """Transition to Simulation Page"""
    # 1. Capture Slider Values
    st.session_state.sim_params["num_agents"] = st.session_state["setup_agents_slider"]
    st.session_state.sim_params["rowdiness_level"] = st.session_state["setup_rowdiness_slider"]
    
    # 2. Ensure a Grid exists (Default if none selected)
    if "structure_grid" not in st.session_state.sim_params:
        st.session_state.sim_params["structure_grid"] = render.get_structure_grid()

    # 3. Switch Page
    st.session_state.sim_running = True
    st.session_state.page = "simulation"

def stop_simulation():
    """Return to Setup Page"""
    st.session_state.sim_running = False
    st.session_state.page = "setup"
    
    # Clear physics state to force a complete reset
    if "physics_state" in st.session_state:
        del st.session_state.physics_state
    if "physics_initialized" in st.session_state:
        del st.session_state.physics_initialized

def reset_optimization_cb():
    """Clear the optimized map and revert to original"""
    st.session_state.is_optimized = False
    # The actual map reload happens in the UI logic below based on selection

# --- PAGE 1: SETUP ---
if st.session_state.page == "setup":
    
    # Layout Containers
    header = st.container()
    controls = st.container()

    with header:
        # 1. Title Area
        logo_c1, logo_c2, logo_c3 = st.columns([1, 3, 1])
        with logo_c2:
            try:
                st.image("assets/titleAndIcon.png", use_container_width=True)
            except:
                st.title("CrowdFlow Simulation")
        
        st.markdown("<h4 style='text-align: center; color: gray;'>Simulate crowded events, save communities</h4>", unsafe_allow_html=True)
        st.markdown("---")

    with controls:
        col1, col2, col3 = st.columns([1, 1, 1])

        # --- LEFT COLUMN: METRICS ---
        with col1:
            st.markdown("### 📡 System Status")
            st.metric(label="Physics Engine", value="Verlet Integration", delta="Active")
            st.metric(label="Optimization Engine", value="Monte Carlo", delta="Ready")
            
            st.markdown("---")
            st.markdown("**Capabilities:**")
            st.markdown("- *Collision Prediction*")
            st.markdown("- *Social Force Model*")
            st.markdown("- *Automated Barrier Placement*")
            st.caption("Build v1.0 [Merged]")

        # --- MIDDLE COLUMN: INPUTS & OPTIMIZER ---
        with col2:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center;'>Configuration</h3>", unsafe_allow_html=True)
                
                # A. Sliders
                st.slider("Number of Agents", 100, 2000, 500, key="setup_agents_slider")
                st.slider("Crowd Rowdiness (Panic)", 0.0, 1.0, 0.0, key="setup_rowdiness_slider")
                
                st.write("---")
                
                # B. Map Selection
                st.subheader("Venue Selection")
                maps_dir = "assets/preset_maps"
                
                # Load available maps
                available_maps = []
                if os.path.exists(maps_dir):
                    available_maps = [f for f in os.listdir(maps_dir) if f.endswith(".csv")]
                
                selected_file = st.selectbox("Choose Layout:", ["Default Arena"] + available_maps)
                
                # Logic to load the map
                current_grid = None
                if selected_file == "Default Arena":
                    # Only reload if we aren't holding an optimized map
                    if not st.session_state.is_optimized:
                        current_grid = render.get_structure_grid(None)
                        st.session_state.sim_params["structure_grid"] = current_grid
                    else:
                        current_grid = st.session_state.sim_params.get("structure_grid")
                else:
                    # CSV Loading
                    if not st.session_state.is_optimized:
                        path = os.path.join(maps_dir, selected_file)
                        current_grid = render.get_structure_grid(path)
                        st.session_state.sim_params["structure_grid"] = current_grid
                    else:
                        current_grid = st.session_state.sim_params.get("structure_grid")

                # C. Map Preview
                if current_grid is not None:
                    # Show badge if optimized
                    if st.session_state.is_optimized:
                        st.success("✅ Map Optimized by AI")
                    
                    preview_img = render.get_preview_image(current_grid)
                    st.image(preview_img, caption="Venue Preview", use_container_width=True)

                # D. OPTIMIZATION BUTTON (The Merged Feature)
                st.write("")
                opt_col1, opt_col2 = st.columns([1, 1])
                
                with opt_col1:
                    if st.button("✨ Auto-Optimise", help="Run AI to find best barrier placement"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        viz_placeholder = st.empty() # Placeholder for the Monte Carlo view
                        
                        # Use the CURRENT map as the baseline
                        base_map = current_grid if current_grid is not None else render.get_structure_grid()
                        
                        # Run the Optimizer
                        best_grid = optimiser.run_optimisation(
                            status_text,
                            viz_placeholder, # Pass placeholder for live drawing
                            st.session_state["setup_agents_slider"],
                            max_iter=100, 
                            patience=20,
                            default_grid=base_map
                        )
                        
                        # Save Result
                        st.session_state.sim_params["structure_grid"] = best_grid
                        st.session_state.is_optimized = True
                        st.rerun() # Rerun to update the preview image
                
                with opt_col2:
                    if st.session_state.is_optimized:
                        st.button("Reset Map", on_click=reset_optimization_cb)

                st.markdown("---")
                
                # E. Start Button
                st.button("🚀 Start Simulation", on_click=start_simulation, type="primary", use_container_width=True)

        # --- RIGHT COLUMN: GUIDE ---
        with col3:
            st.markdown("### 📋 User Guide")
            with st.expander("How to use", expanded=True):
                st.markdown("""
                1. **Setup**: Choose agent count and panic level.
                2. **Venue**: Select a map from the dropdown.
                3. **Optimise**: Click 'Auto-Optimise' to let the AI place barriers for you.
                4. **Run**: Click Start to watch the physics live.
                """)
            st.info("💡 **Tip:** Optimisation runs a fast 'headless' simulation to test hundreds of barrier layouts.")

# --- PAGE 2: SIMULATION ---
elif st.session_state.page == "simulation":
    
    # 1. Retrieve Config
    init_agents = st.session_state.sim_params.get("num_agents", 500)
    init_rowdiness = st.session_state.sim_params.get("rowdiness_level", 0.0)
    structure_grid = st.session_state.sim_params.get("structure_grid")

    # Safety Check
    if structure_grid is None:
        st.error("Map data missing. Returning to setup...")
        if st.button("Back"): stop_simulation()
    else:
        # 2. Sidebar Controls (Live Updates)
        rowdiness = render.render_sidebar_controls(
            stop_callback=stop_simulation, 
            initial_rowdiness=init_rowdiness
        )

        # 3. Main Simulation Area
        st.markdown("<h2 style='text-align: center;'>Live Simulation</h2>", unsafe_allow_html=True)
        
        # Center the canvas using columns
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            canvas_placeholder = st.empty()
        
        # 4. Run the Game Loop
        loop.run_simulation(canvas_placeholder, init_agents, rowdiness, structure_grid)
