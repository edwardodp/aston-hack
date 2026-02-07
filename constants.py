# --- LOGICAL WORLD (The Physics Contract) ---
LOGICAL_WIDTH = 1024.0
LOGICAL_HEIGHT = 1024.0
GRID_ROWS = 32
GRID_COLS = 32

# --- RENDERING (The Display Contract) ---
CANVAS_SIZE_PX = 612
PIXELS_PER_CELL = int(CANVAS_SIZE_PX / GRID_COLS) # 20px

# --- ENTITY IDs ---
ID_NOTHING = 0
ID_WALL = 1
ID_POI = 2
ID_BARRIER = 3

# --- COLORS (BGR format for OpenCV) ---
COLOR_BG           = (255, 255, 255) # White
COLOR_WALL         = (64, 64, 64)    # Dark Gray
COLOR_GRID_LINE    = (200, 200, 200) # Light Gray
COLOR_AGENT_SAFE   = (0, 255, 0)     # Green
COLOR_AGENT_WARN   = (0, 165, 255)   # Orange
COLOR_AGENT_DANGER = (0, 0, 255)     # Red

# --- PHYSICS ---
DT = 0.05
SUB_TICKS = 2