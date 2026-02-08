# --- LOGICAL WORLD (The Physics Contract) ---
LOGICAL_WIDTH = 1024.0
LOGICAL_HEIGHT = 1024.0
GRID_ROWS = 32
GRID_COLS = 32

# --- RENDERING (The Display Contract) ---
CANVAS_SIZE_PX = 640
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
COLOR_POI      = (100, 0, 100)       # Dark Purple
COLOR_BARRIER  = (0, 215, 255)       # Dark Yellow/Gold (B=0, G=215, R=255)

# --- PHYSICS ENGINE CONFIG ---
DT = 0.016         # Time step (seconds)
SUB_TICKS = 1     # How many physics steps per 1 render frame

# --- CORE PHYSICS CONSTANTS ---
BASE_MASS = 3.0              # kg (Mass of an average agent)
REF_DIAMETER = 23.0          # px (Reference size for mass scaling)

# Forces
WALL_STIFFNESS = 5000.0      # k (Stronger spring to prevent wall clipping)
AGENT_CONTACT_STIFFNESS = 1000.0 # k (Spring force between agents to stop overlap)

WALL_REPULSION_DIST = 5.0    # meters (Soft warning zone)
WALL_REPULSION_STRENGTH = 1000.0

# --- PERSONALITY PROFILES (Rowdiness Map) ---
# 1. Desired Speed
SPEED_CALM = 80.0
SPEED_PANIC = 400.0

# 2. Reaction Time
TAU_CALM = 0.8
TAU_PANIC = 0.1

# 3. Personal Space (Psychological Radius)
# Note: This is separate from Physical Radius (Diameter/2)
RADIUS_CALM = 100.0
RADIUS_PANIC = 30.0

# 4. Social Repulsion (Psychological Strength)
SOCIAL_PUSH_CALM = 50.0
SOCIAL_PUSH_PANIC = 1000.0

# 5. Noise
NOISE_CALM = 5.0
NOISE_PANIC = 500.0
