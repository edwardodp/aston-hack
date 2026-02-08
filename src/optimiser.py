import numpy as np
import random
import cv2
import streamlit as st
from . import physics
from . import render
from . import strategies

# --- 1. EVALUATION ENGINE ---

def evaluate_layout(grid_layout, num_agents, rowdiness=1.0, steps=350):
    """
    Runs a headless simulation to score a layout.
    """
    state = physics.init_state(num_agents, grid_layout)
    total_pressure = 0.0
    
    for _ in range(steps):
        # Pass the specific rowdiness to the physics engine
        state = physics.tick(state, rowdiness, grid_layout)
        
        # USE THE SINGLE SOURCE OF TRUTH for the metric
        total_pressure += physics.get_average_pressure(state)
        
    return total_pressure / steps

# --- 2. MAIN OPTIMISATION LOOP ---

def run_optimisation(status_placeholder, canvas_placeholder, num_agents, rowdiness=1.0, max_iter=300, patience=30, default_grid=None):
    """
    Optimiser with 'Patience' and Active Visualisation.
    """
    
    # 1. Benchmarking Baseline
    if default_grid is None:
        default_grid = render.get_structure_grid()
        
    status_placeholder.info(f"Benchmarking Default Map (Rowdiness: {rowdiness*100:.0f}%)...")
    
    # Visualise Baseline
    dummy_pos = np.zeros((0, 2))
    base_frame = render.render_frame(dummy_pos, np.zeros(0), np.zeros(0), default_grid)
    canvas_placeholder.image(base_frame, channels="RGB", caption="Baseline: Default Layout")
    
    # Evaluate Baseline with user-defined rowdiness
    baseline_score = evaluate_layout(default_grid, num_agents, rowdiness)
    
    best_grid = default_grid
    best_score = baseline_score
    best_desc = "Default Layout"
    
    status_placeholder.write(f"**Baseline Score:** {baseline_score:.2f}")
    
    # 2. Smart Search Loop
    attempts_since_last_improvement = 0
    
    for i in range(max_iter):
        # Check Patience
        if attempts_since_last_improvement >= patience:
            status_placeholder.warning(f"Stopping early: No improvement for {patience} steps.")
            break

        # A. Pick & Generate Strategy
        strategy_func = random.choice(strategies.ALL_STRATEGIES)
        clean_map = strategies.get_static_map()
        candidate_grid, desc = strategy_func(clean_map)
        
        # --- B. SPEED VISUALISATION ---
        # Render the candidate grid to an image
        preview_frame = render.render_frame(dummy_pos, np.zeros(0), np.zeros(0), candidate_grid)
        
        # Draw Iteration Number on the image (Black Text)
        cv2.putText(preview_frame, f"Iter {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
        # Send to the Main Canvas (No Sleep = Fast/Zipping view)
        canvas_placeholder.image(preview_frame, channels="RGB", caption=f"Testing: {desc}")
        # ---------------------------------
        
        # C. Evaluate with user-defined rowdiness
        score = evaluate_layout(candidate_grid, num_agents, rowdiness)
        
        # D. Compare
        if score < best_score:
            best_score = score
            best_grid = candidate_grid
            best_desc = desc
            attempts_since_last_improvement = 0 
            status_placeholder.success(f"New Best! {desc} (Score: {best_score:.2f})")
        else:
            attempts_since_last_improvement += 1
            status_placeholder.text(f"Iter {i+1}: {desc} (Score {score:.2f}) - Patience {patience - attempts_since_last_improvement} left")
            
    # Final Result
    status_placeholder.success(f"Optimisation Complete! Winner: {best_desc}")
    st.caption(f"Final Score: {best_score:.2f} (Improvement: {baseline_score - best_score:.2f})")
    
    return best_grid
