import streamlit as st
from PIL import Image
from src import render
from src import loop
from src import optimiser
from src import music_analyser
import tempfile
import os


try:
    favicon_image = Image.open("assets/crowd.png")
    st.set_page_config(page_title="Crowd Flow", page_icon=favicon_image, layout="wide")
except:
    st.set_page_config(page_title="Crowd Flow", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "setup"

if "sim_params" not in st.session_state:
    st.session_state.sim_params = {}

if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

if "is_optimized" not in st.session_state:
    st.session_state.is_optimized = False

def start_simulation():
    st.session_state.sim_params["num_agents"] = st.session_state["setup_agents_slider"]
    st.session_state.sim_params["rowdiness_level"] = st.session_state["setup_rowdiness_slider"]
    st.session_state.sim_params["switch_chance"] = st.session_state["setup_switch_slider"]
    
    if not st.session_state.is_optimized:
        maps_dir = "assets/preset_maps"
        selected_file = st.session_state.get("setup_map_select")
        
        if selected_file and os.path.exists(os.path.join(maps_dir, selected_file)):
            path = os.path.join(maps_dir, selected_file)
            grid = render.get_structure_grid(path)
        else:
            grid = render.get_structure_grid(None)
            
        st.session_state.sim_params["structure_grid"] = grid

    st.session_state.page = "simulation"
    st.session_state.sim_running = True

def stop_simulation():
    st.session_state.sim_running = False
    st.session_state.page = "setup"
    
    if "physics_state" in st.session_state:
        del st.session_state.physics_state
    
    if not st.session_state.is_optimized and "structure_grid" in st.session_state.sim_params:
        del st.session_state.sim_params["structure_grid"]

def reset_optimization_cb():
    st.session_state.is_optimized = False
    if "structure_grid" in st.session_state.sim_params:
        del st.session_state.sim_params["structure_grid"]

main_interface = st.empty()

if st.session_state.page == "setup":
    
    with main_interface.container():
        
        logo_c1, logo_c2, logo_c3 = st.columns([1, 3, 1])
        with logo_c2:
            try:
                st.image("assets/titleAndIcon.png", use_container_width=True)
            except:
                st.title("Crowd Flow")
        
        st.markdown("<h4 style='text-align: center; color: gray;'>Simulate crowded events, save communities</h4>", unsafe_allow_html=True)
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("### System Status")
            st.metric(label="Physics Engine", value="Verlet", delta="60 FPS")
            st.metric(label="Optimisation Engine", value="Monte Carlo", delta="Working")
            st.markdown("---")
            st.markdown("**Capabilities:**")
            st.markdown("- *Collision Prediction*")
            st.markdown("- *Social Force Model*")
            st.markdown("- *Automated Barrier Placement*")
            st.markdown("- *Real time pressure graph*")
            st.markdown("- *Audio Integration with Model*")
        
        with col2:
            with st.container(border=True):
                
                @st.fragment
                def render_setup_interface():
                    st.markdown("<h3 style='text-align: center;'>Configuration</h3>", unsafe_allow_html=True)
                    
                    has_music = "music_data" in st.session_state.sim_params
                    
                    st.slider("Number of Agents", 1, 2000, 250, key="setup_agents_slider")
                    st.write("") 
                    st.slider(
                        "Crowd Rowdiness", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.0, 
                        key="setup_rowdiness_slider",
                        disabled=has_music,
                        help="Locked when Music Integration is active." if has_music else "Simulate a more pushy and agitated crowd."
                    )
                    st.write("")
                    st.slider(
                        "Wandering", 
                        min_value=0.0, 
                        max_value=0.005, 
                        value=0.001, 
                        step=0.0001,
                        format="%.4f",
                        key="setup_switch_slider",
                        disabled=has_music,
                        help="Locked when Music Integration is active." if has_music else "Probability per tick that an agent changes destination."
                    )
                    st.write("")
                    
                    st.subheader("Venue Selection")
                    maps_dir = "assets/preset_maps"
                    available_maps = []
                    if os.path.exists(maps_dir):
                        available_maps = [f for f in os.listdir(maps_dir) if f.endswith(".csv")]
                    
                    selected_file = st.selectbox(
                        "Choose Layout:",    
                        available_maps if available_maps else ["Default"],
                        key="setup_map_select"
                    )

                    if st.session_state.is_optimized:
                        current_grid = st.session_state.sim_params.get("structure_grid")
                        
                    else:
                        if available_maps and selected_file in available_maps:
                            path = os.path.join(maps_dir, selected_file)
                            current_grid = render.get_structure_grid(path)
                        else:
                            current_grid = render.get_structure_grid(None)
                    

                    c1, c2 = st.columns([1, 2])
                    
                    with c2:
                        preview_image = None 
                        if current_grid is not None:
                            preview_image = render.get_preview_image(current_grid)
                        
                        if preview_image is not None:
                            rendered_image = st.image(preview_image, caption="Venue Preview", use_container_width=True)
                        else:
                            st.info("No preview available for this map.")
                            rendered_image = st.empty()
                    
                    with c1:
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
                        
                        st.write("")
                        if not st.session_state.is_optimized:
                            if st.button("Auto-Optimise", help="Run AI to find best barrier placement"):
                                status_text = st.empty()
                                
                                viz_placeholder = rendered_image
                                
                                best_grid = optimiser.run_optimisation(
                                    status_placeholder=status_text,
                                    canvas_placeholder=viz_placeholder,
                                    num_agents=st.session_state["setup_agents_slider"],
                                    rowdiness=st.session_state["setup_rowdiness_slider"],
                                    max_iter=1000,
                                    patience=200,
                                    default_grid=current_grid
                                )
                                
                                st.session_state.sim_params["structure_grid"] = best_grid
                                st.session_state.is_optimized = True
                                st.rerun()
                                
                        if st.session_state.is_optimized:
                            if st.button("Reset Map"):
                                reset_optimization_cb()
                                st.rerun()
                
                
                st.write("")
                
                render_setup_interface()
                
                st.subheader("Audio Integration")
                music_upload = st.file_uploader("Upload Music (mp3/wav)", type=['mp3', 'wav', 'ogg'])

                if music_upload is None:
                    if "music_data" in st.session_state.sim_params:
                        del st.session_state.sim_params["music_data"]
                        del st.session_state.sim_params["music_file"]
                        st.rerun()
                
                elif music_upload is not None:
                    if "music_data" not in st.session_state.sim_params:
                        if st.button("Analyze Audio"):
                            with st.spinner("Analyzing RMS Energy, Onset Strength and Tempo..."):
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(music_upload.name)[1]) as tmp:
                                    tmp.write(music_upload.getvalue())
                                    tmp_path = tmp.name
                                
                                try:
                                    data = music_analyser.analyze_music_for_rowdiness(tmp_path)
                                    st.session_state.sim_params["music_data"] = data
                                    st.session_state.sim_params["music_file"] = music_upload
                                    st.rerun() # Refresh to lock sliders
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                finally:
                                    if os.path.exists(tmp_path): os.remove(tmp_path)
                    else:
                        st.success(f"Analysis Complete! ({st.session_state.sim_params['music_data']['tempo']:.1f} BPM)")
                        st.info("🔒 Rowdiness & Wandering are now controlled by the music.")

                
                st.markdown("---")
                
                st.button("Start Simulation", on_click=start_simulation, type="primary", use_container_width=True)

        with col3:
            st.markdown("### User Guide")
            with st.expander("How to use", expanded=True):
                st.markdown("""
                1. **Setup**: Choose agent count and panic level.
                2. **Venue**: Select a map from the dropdown.
                3. **Optimise**: Click 'Auto-Optimise' to let the AI place barriers.
                4. **Run**: Watch the crowd vibe to the beat with live physics.
                """)
            st.info("💡 **Tip:** Optimisation runs a fast 'headless' simulation to test hundreds of layouts.")

elif st.session_state.page == "simulation":
    
    main_interface.empty()
    
    init_agents = st.session_state.sim_params.get("num_agents", 500)
    init_rowdiness = st.session_state.sim_params.get("rowdiness_level", 0.0)
    init_switch = st.session_state.sim_params.get("switch_chance", 0.001)
    structure_grid = st.session_state.sim_params.get("structure_grid")
    
    music_active = "music_data" in st.session_state.sim_params

    if structure_grid is None:
        st.error("No map data found. Please return to setup.")
        if st.button("Back to Setup"):
            stop_simulation()
            st.rerun()
    else:
        rowdiness, switch_chance, chart_placeholder = render.render_sidebar_controls(
            stop_callback=stop_simulation, 
            initial_rowdiness=init_rowdiness,
            initial_switch_chance=init_switch,
            music_active=music_active
        )

        st.markdown("<h2 style='text-align: center;'>Live Simulation</h2>", unsafe_allow_html=True)
        
        music_file = st.session_state.sim_params.get("music_file")
        if music_file:
            st.audio(music_file, format=music_file.type, autoplay=True)
            st.caption(f"Now Playing: {music_file.name}")
        
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            canvas_placeholder = st.empty()
        
        loop.run_simulation(
            canvas_placeholder, 
            init_agents, 
            rowdiness, 
            structure_grid, 
            switch_chance,
            chart_placeholder
        )
