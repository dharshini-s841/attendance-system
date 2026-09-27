import os

# --- Cloud backend ---
CLOUD_API_URL = os.environ.get("CLOUD_API_URL", "http://localhost:5000")
CLOUD_API_KEY = os.environ.get("CLOUD_API_KEY", "change-me")  # sent as header for auth

# --- Camera / classroom identity ---
CLASSROOM_ID = os.environ.get("CLASSROOM_ID", "room-101")

# --- dlib landmark model ---
SHAPE_PREDICTOR_PATH = os.path.join(
    os.path.dirname(__file__), "models", "shape_predictor_68_face_landmarks.dat"
)

# --- Fixed periods (24h time). Add as many as needed. ---
PERIODS = [
    {"name": "Period 1", "start": "09:00", "end": "09:50"},
    {"name": "Period 2", "start": "10:00", "end": "10:50"},
    {"name": "Period 3", "start": "11:00", "end": "11:50"},
]

# --- Recognition / liveness tuning ---
PROCESS_EVERY_N_FRAMES = 3  # skip frames to reduce CPU load
BLINK_WINDOW_SECONDS = 6           # time allowed to detect a blink after a match
FRAME_RESIZE_SCALE = 1.0 # downscale frames before detection for speed
