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

# --- 2. SAFE PLACEMENT HELPER ---

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
    # (We iterate to handle potential walls in the middle of a line)
    for col in range(col_start, col_end):
        if grid[row, col] == c.ID_NOTHING:
            grid[row, col] = c.ID_BARRIER

# --- 3. BARRIER PATTERNS (UNBIASED) ---

def strategy_single_row(grid):
    """
    A single horizontal barrier.
    RANGE: Anywhere vertically (Top to Bottom).
    """
    # 1. Randomize Position (Anywhere in the arena)
    row = np.random.randint(2, c.GRID_ROWS - 2)
    
    # 2. Randomize Gap
    gap = np.random.randint(2, 8)
    
    # 3. Randomize Horizontal Center of the gap (Not just the middle of map)
    gap_center = np.random.randint(4, c.GRID_COLS - 4)
    
    left_end = gap_center - (gap // 2)
    right_start = gap_center + (gap // 2)
    
    # 4. Apply
    safe_place_barrier(grid, row, 1, left_end)
    safe_place_barrier(grid, row, right_start, c.GRID_COLS - 1)
    
    return grid, f"Single Row (Row {row})"

def strategy_zigzag(grid):
    """
    Two staggered rows.
    RANGE: Anywhere vertically.
    """
    # Top Row
    row_top = np.random.randint(2, c.GRID_ROWS - 6)
    
    # Bottom Row (2-6 rows below top)
    row_bot = row_top + np.random.randint(2, 6)
    
    # Randomize gap positions independently
    gap_top = np.random.randint(2, c.GRID_COLS - 10)
    gap_bot = np.random.randint(2, c.GRID_COLS - 10)
    
    # Apply Top (Block Right side)
    safe_place_barrier(grid, row_top, gap_top, c.GRID_COLS - 1)
    
    # Apply Bottom (Block Left side)
    safe_place_barrier(grid, row_bot, 1, gap_bot)
    
    return grid, f"Zig-Zag (Rows {row_top}/{row_bot})"

def strategy_lane_split(grid):
    """
    Vertical divider.
    RANGE: Anywhere horizontally (Left to Right).
    """
    # Random Length
    length = np.random.randint(5, 20)
    
    # Random Column (Anywhere from left to right)
    col = np.random.randint(2, c.GRID_COLS - 2)
    
    # Random Start Row
    start_row = np.random.randint(2, c.GRID_ROWS - length - 2)
    end_row = start_row + length
    
    # Vertical Safe Placement
    for r in range(start_row, end_row):
        if grid[r, col] == c.ID_NOTHING:
            grid[r, col] = c.ID_BARRIER
    
    return grid, f"Vertical Split (Col {col})"

def strategy_checkerboard(grid):
    """
    Random Islands.
    RANGE: Entire Grid.
    """
    # Create a grid of potential island spots
    active_count = 0
    
    # Try 10 random islands
    for _ in range(10):
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
                
    return grid, f"Islands (Count {active_count})"

def strategy_funnel_entry(grid):
    """
    V-Shape Funnel.
    RANGE: Anywhere (Can act as a wedge/plow).
    """
    # Random Tip Location
    tip_row = np.random.randint(4, c.GRID_ROWS - 10)
    tip_col = np.random.randint(5, c.GRID_COLS - 5)
    
    width = np.random.randint(3, 6)
    
    for i in range(width):
        # Left wing
        r, c_idx = tip_row + i, tip_col - i - 1
        if 0 <= r < c.GRID_ROWS and 0 <= c_idx < c.GRID_COLS:
            if grid[r, c_idx] == c.ID_NOTHING:
                grid[r, c_idx] = c.ID_BARRIER
                
        # Right wing
        r, c_idx = tip_row + i, tip_col + i + 1
        if 0 <= r < c.GRID_ROWS and 0 <= c_idx < c.GRID_COLS:
            if grid[r, c_idx] == c.ID_NOTHING:
                grid[r, c_idx] = c.ID_BARRIER
        
    return grid, f"Wedge (Tip {tip_row},{tip_col})"

# --- EXPORT LIST ---
ALL_STRATEGIES = [
    strategy_single_row, 
    strategy_zigzag, 
    strategy_lane_split,
    strategy_checkerboard,
    strategy_funnel_entry
]
