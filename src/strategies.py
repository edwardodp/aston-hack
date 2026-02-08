import numpy as np
from . import constants as c
from . import render 

# --- 1. THE VENUE (Static Base) ---

def get_static_map():
    """
    Returns the base venue directly from the Render module.
    Removes default barriers to ensure a clean slate for optimization.
    """
    # 1. Fetch the master grid
    grid = render.get_structure_grid().copy()
    
    # 2. Clean Slate: Remove any pre-existing barriers (ID=3)
    grid[grid == c.ID_BARRIER] = c.ID_NOTHING
        
    return grid

# --- 2. SAFE PLACEMENT HELPERS ---

def safe_place_barrier(grid, row, col_start, col_end):
    """
    Helper to place a horizontal barrier segment safely.
    It ONLY overwrites Empty cells (ID_NOTHING).
    """
    # Ensure indices are within bounds
    row = int(np.clip(row, 0, c.GRID_ROWS - 1))
    col_start = int(np.clip(col_start, 0, c.GRID_COLS))
    col_end = int(np.clip(col_end, 0, c.GRID_COLS))
    
    # Iterate and place carefully
    for col in range(col_start, col_end):
        if grid[row, col] == c.ID_NOTHING:
            grid[row, col] = c.ID_BARRIER

def safe_place_barrier_vertical(grid, col, row_start, row_end):
    """
    Helper to place a VERTICAL barrier segment safely.
    It ONLY overwrites Empty cells (ID_NOTHING).
    """
    # Ensure indices are within bounds
    col = int(np.clip(col, 0, c.GRID_COLS - 1))
    row_start = int(np.clip(row_start, 0, c.GRID_ROWS))
    row_end = int(np.clip(row_end, 0, c.GRID_ROWS))
    
    # Iterate and place carefully
    for row in range(row_start, row_end):
        if grid[row, col] == c.ID_NOTHING:
            grid[row, col] = c.ID_BARRIER

# --- 3. MAIN STRATEGIES ---

def strategy_empty(grid):
    """
    Places NO main barriers. 
    Allows the optimizer to test 'Sub-Strategies Only' (Just Cordon, Just Hairs).
    """
    return _apply_sub_strategies(grid, "No Main Barriers")

def strategy_row_interweaved(grid):
    """
    Creates ROWS of barriers with varied lengths (1-6) interweaved.
    """
    bar_len = np.random.randint(1, 7)
    gap_len = np.random.randint(2, 6)
    v_spacing = np.random.randint(3, 8)
    
    period = bar_len + gap_len
    row_offset = period // 2
    start_row = np.random.randint(3, 8)
    
    row_count = 0
    for r in range(start_row, c.GRID_ROWS - 3, v_spacing):
        current_shift = 0 if (row_count % 2 == 0) else row_offset
        for col in range(current_shift, c.GRID_COLS, period):
            c_end = col + bar_len
            safe_place_barrier(grid, r, col, c_end)
        row_count += 1
        
    desc = f"Row Interweaved (L={bar_len}, Gap={gap_len})"
    return _apply_sub_strategies(grid, desc)

def strategy_col_interweaved(grid):
    """
    Creates COLUMNS of barriers with varied lengths (1-6) interweaved.
    """
    bar_len = np.random.randint(1, 7)
    gap_len = np.random.randint(2, 6)
    h_spacing = np.random.randint(3, 8)
    
    period = bar_len + gap_len
    col_offset = period // 2
    start_col = np.random.randint(3, 8)
    
    col_count = 0
    for col_idx in range(start_col, c.GRID_COLS - 3, h_spacing):
        current_shift = 0 if (col_count % 2 == 0) else col_offset
        for r in range(current_shift, c.GRID_ROWS, period):
            r_end = r + bar_len
            safe_place_barrier_vertical(grid, col_idx, r, r_end)
        col_count += 1
        
    desc = f"Col Interweaved (L={bar_len}, Gap={gap_len})"
    return _apply_sub_strategies(grid, desc)

def strategy_islands(grid):
    """
    Places random 2x2 'pillars' or islands to break up flow.
    Good for disrupting waves without blocking paths.
    """
    num_islands = np.random.randint(5, 15)
    active_count = 0
    for _ in range(num_islands):
        r = np.random.randint(2, c.GRID_ROWS - 3)
        c_idx = np.random.randint(2, c.GRID_COLS - 3)
        
        # Place a 2x2 island safely
        placed = False
        for dr in range(2):
            for dc in range(2):
                if grid[r+dr, c_idx+dc] == c.ID_NOTHING:
                    grid[r+dr, c_idx+dc] = c.ID_BARRIER
                    placed = True
        if placed:
            active_count += 1
                
    return _apply_sub_strategies(grid, f"Islands (Count {active_count})")

def strategy_multi_funnels(grid):
    """
    Places multiple V-shape Wedges (Funnels) of various orientations.
    Acts as a field of 'plows' to split crowds in multiple directions.
    """
    num_funnels = np.random.randint(4, 9) # 4 to 8 funnels
    
    for _ in range(num_funnels):
        # Random Center
        r = np.random.randint(5, c.GRID_ROWS - 5)
        c_idx = np.random.randint(5, c.GRID_COLS - 5)
        
        # Random Size & Orientation
        width = np.random.randint(3, 7)
        orientation = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        _place_wedge(grid, r, c_idx, width, orientation)
        
    return _apply_sub_strategies(grid, f"Multi-Funnels (x{num_funnels})")

def _place_wedge(grid, r, c_idx, width, orientation):
    """
    Draws a V-shape barrier.
    """
    for i in range(width):
        # Calculate wing offsets based on 'i' (distance from tip)
        
        # OFFSETS (dr, dc) for Left Wing and Right Wing
        if orientation == 'UP':    # ^ Shape (Points Up, Wings go Down/Out)
            w1 = (i, -i); w2 = (i, i)
        elif orientation == 'DOWN': # v Shape (Points Down, Wings go Up/Out)
            w1 = (-i, -i); w2 = (-i, i)
        elif orientation == 'LEFT': # < Shape (Points Left, Wings go Right/Out)
            w1 = (-i, i); w2 = (i, i)
        elif orientation == 'RIGHT': # > Shape (Points Right, Wings go Left/Out)
            w1 = (-i, -i); w2 = (i, -i)
            
        # Draw Wings
        for dr, dc in [w1, w2]:
            nr, nc = r + dr, c_idx + dc
            if 0 <= nr < c.GRID_ROWS and 0 <= nc < c.GRID_COLS:
                if grid[nr, nc] == c.ID_NOTHING:
                    grid[nr, nc] = c.ID_BARRIER

# --- 4. SUB-STRATEGIES (Add-ons) ---

def strategy_poi_cordon(grid):
    """
    Places a protective layer of barriers around all POI cells.
    """
    poi_rows, poi_cols = np.where(grid == c.ID_POI)
    if len(poi_rows) == 0: return grid, ""

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    applied = False
    for r, col in zip(poi_rows, poi_cols):
        for dr, dc in neighbor_offsets:
            nr, nc = r + dr, col + dc
            if 0 <= nr < c.GRID_ROWS and 0 <= nc < c.GRID_COLS:
                if grid[nr, nc] == c.ID_NOTHING:
                    grid[nr, nc] = c.ID_BARRIER
                    applied = True
    return grid, " + POI Cordon" if applied else ""

def strategy_poi_hairs(grid):
    """
    Creates 'rods' (hairs) of length 3 extending from POI surfaces.
    Uses structured spacing (every 3rd cell).
    """
    poi_rows, poi_cols = np.where(grid == c.ID_POI)
    if len(poi_rows) == 0: return grid, ""
    
    spacing = 3; hair_length = 3; applied_count = 0
    
    for r, col in zip(poi_rows, poi_cols):
        # North
        if r > 0 and grid[r-1, col] == c.ID_NOTHING and col % spacing == 0:
            _grow_hair(grid, r, col, -1, 0, hair_length); applied_count += 1
        # South
        if r < c.GRID_ROWS - 1 and grid[r+1, col] == c.ID_NOTHING and col % spacing == 0:
            _grow_hair(grid, r, col, 1, 0, hair_length); applied_count += 1
        # West
        if col > 0 and grid[r, col-1] == c.ID_NOTHING and r % spacing == 0:
            _grow_hair(grid, r, col, 0, -1, hair_length); applied_count += 1
        # East
        if col < c.GRID_COLS - 1 and grid[r, col+1] == c.ID_NOTHING and r % spacing == 0:
            _grow_hair(grid, r, col, 0, 1, hair_length); applied_count += 1

    return grid, " + POI Hairs" if applied_count > 0 else ""

def _grow_hair(grid, r, c_idx, dr, dc, length):
    for k in range(1, length + 1):
        nr, nc = r + (dr * k), c_idx + (dc * k)
        if 0 <= nr < c.GRID_ROWS and 0 <= nc < c.GRID_COLS:
            if grid[nr, nc] == c.ID_NOTHING:
                grid[nr, nc] = c.ID_BARRIER
            else: break

# List of available sub-strategies
SUB_STRATEGIES = [strategy_poi_cordon, strategy_poi_hairs]

def _apply_sub_strategies(grid, current_desc):
    """
    Randomly selects ONE sub-strategy based on chance.
    """
    if np.random.random() < c.SUB_STRATEGY_CHANCE:
        strat_idx = np.random.randint(len(SUB_STRATEGIES))
        sub_strat = SUB_STRATEGIES[strat_idx]
        grid, sub_desc = sub_strat(grid)
        current_desc += sub_desc
    return grid, current_desc

# --- EXPORT LIST ---
ALL_STRATEGIES = [
    strategy_empty,
    strategy_row_interweaved,
    strategy_col_interweaved,
    strategy_islands,
    strategy_multi_funnels  # UPDATED
]
