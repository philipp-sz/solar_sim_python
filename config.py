"""Fixed settings of the simulation: paths, window, meshes, camera, time stepping. Per-body data lives in calculations/bodies.py."""
from pathlib import Path


# --- Paths (anchored at this file, so the program works from any working directory) ---
PROJECT_DIR = Path(__file__).parent
SHADER_DIR = PROJECT_DIR / "shaders"  # <name>.vert and <name>.frag per GPU program
TEXTURE_QUALITY = "2k"  # "2k" (in the repository) or "8k" (downloaded, assets/textures_8k, not in the repository, see CREDITS.txt)
TEXTURE_DIRS = {"2k": PROJECT_DIR / "assets" / "textures", "8k": PROJECT_DIR / "assets" / "textures_8k"}
TEXTURE_DIR = TEXTURE_DIRS[TEXTURE_QUALITY]  # both folders have the same structure
if not TEXTURE_DIR.is_dir():
    raise FileNotFoundError(f"TEXTURE_QUALITY is '{TEXTURE_QUALITY}', but {TEXTURE_DIR} does not exist (see assets/textures/CREDITS.txt)")
BODY_TEXTURE_DIR = TEXTURE_DIR / "bodies"  # one folder per body: <name>/NN_<variant>.jpg
RING_TEXTURE_PATH = TEXTURE_DIR / "rings" / "saturn.png"
CLOUD_TEXTURE_PATH = TEXTURE_DIR / "overlays" / "earth_clouds.jpg"
SKY_TEXTURE_PATH = TEXTURE_DIR / "sky" / "milky_way.jpg"
SKY_BRIGHTNESS = 1.1  # the 8K map is darker on average than the 2K map

# --- Window ---
WINDOW_TITLE = "Solar System Simulation"
WINDOW_SIZE = (800, 600)  # pixels
GL_VERSION = (3, 3)
BACKGROUND_COLOR = (0.1, 0.1, 0.1, 1.0)  # RGBA, values 0..1

# --- Meshes ---
SPHERE_STACKS = 32  # horizontal bands of the UV sphere
SPHERE_SLICES = 64  # vertical slices of the UV sphere
RING_BODY_NAME = "Saturn"  # body that carries the ring
RING_SEGMENTS = 64
RING_INNER = 0.495  # inner / outer radius of the ring, derived from the extent of saturn.png (NOTES.md 3.15)
RING_SCALE = 2.42  # outer ring radius in radii of the body (140 850 km / 58 232 km)

# --- Texture units (the shader samplers read from these slots) ---
TEXTURE_UNIT_MAIN = 0  # surface map of the current body, or the ring texture
TEXTURE_UNIT_NIGHT = 1  # night map of the Earth
TEXTURE_UNIT_CLOUDS = 2  # cloud layer of the Earth
TEXTURE_UNIT_RING = 3  # ring texture while the bodies are drawn (ring shadow on the planet)

# --- Camera ---
CAMERA_START_AZIMUTH = 33.69  # degrees, angle around the z axis
CAMERA_START_ELEVATION = 47.97  # degrees, angle above the xy plane
CAMERA_START_DIST = 10.0  # AU
CAMERA_MAX_DIST = 100.0  # AU, limit for zooming out
CAMERA_MAX_ELEVATION = 89.0  # degrees, stays below 90 so that look_at has a defined right vector
CAMERA_UP = (0.0, 0.0, 1.0)  # ecliptic north
CAMERA_FOV = 60.0  # degrees, vertical field of view
CAMERA_NEAR_FACTOR = 0.01  # near plane = factor * distance to the target
CAMERA_FAR = 200.0  # AU
MOUSE_SENSITIVITY = 0.01  # radians per pixel of mouse movement
ZOOM_STEP = 1.1  # distance factor per mouse wheel notch

# --- Time stepping and simulation ---
DT = 1.0  # days per integration step
TIME_FACTOR = 60.0  # simulated days per real second
MAX_STEPS_PER_FRAME = 200  # protects against a frozen frame causing a long queue of steps
TRAIL_POINTS = 200  # points per orbit trail
TIME_FACTOR_MIN = 30.0
TIME_FACTOR_MAX = 1920.0

# --- Rotation ---
SPIN_CAP_ENABLED = True  # limit the visible rotation speed (avoids aliasing at high time factors)
MAX_SPIN_RATE = 360.0  # degrees per real second
INSPECT_SPIN_RATE = 20.0  # degrees per real second while the space bar is held
