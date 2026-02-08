import numpy as np
import time
import random
import cv2
import streamlit as st
from . import physics
from . import render
from . import strategies

# --- 1. EVALUATION ENGINE ---

def evaluate_layout(grid_layout, num_agents, steps=1000):
    """
    Runs a headless simulation.
    """
    state = physics.init_state(num_agents, grid_layout)
    total_pressure = 0.0
    
    for _ in range(steps):
        state = physics.tick(state, 1.0, grid_layout)
        # USE THE SINGLE SOURCE OF TRUTH for the metric
        total_pressure += physics.get_average_pressure(state)
        
    return total_pressure / steps

# --- 2. MAIN OPTIMISATION LOOP ---

def run_optimisation(status_placeholder, canvas_placeholder, num_agents, max_iter=100, patience=10, default_grid=None):
    """
    Optimiser with 'Patience' and Active Visualisation.
    """
    
    # 1. Benchmarking Baseline
    if default_grid is None:
        default_grid = render.get_structure_grid()
        
    status_placeholder.info("Benchmarking Default Map... Please wait.")
    
    # Visualise Baseline
    dummy_pos = np.zeros((0, 2))
    base_frame = render.render_frame(dummy_pos, np.zeros(0), np.zeros(0), default_grid)
    canvas_placeholder.image(base_frame, channels="RGB", caption="Baseline: Default Layout")
    
    baseline_score = evaluate_layout(default_grid, num_agents)
    
    best_grid = default_grid
    best_score = baseline_score
    best_desc = "Default Layout"
    
    status_placeholder.write(f"**Baseline Score:** {baseline_score:.2f}")
    time.sleep(0.5)
    
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
        
        # --- B. RESTORED VISUALISATION ---
        # Render the candidate grid to an image
        preview_frame = render.render_frame(dummy_pos, np.zeros(0), np.zeros(0), candidate_grid)
        
        # Draw Iteration Number on the image (Black Text)
        cv2.putText(preview_frame, f"Iter {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
        # Send to the Main Canvas
        canvas_placeholder.image(preview_frame, channels="RGB", caption=f"Testing: {desc}")
        
        # Small delay so the human eye can see what's happening
        time.sleep(0.1) 
        # ---------------------------------
        
        # C. Evaluate
        score = evaluate_layout(candidate_grid, num_agents)
        
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
