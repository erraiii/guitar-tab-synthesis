from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MAX_HANDS = 2

# VIDEO_PATH = "C:/Users/aivan/Downloads/oceans.mp4"

AUDIO_SERVICE_URL = "http://localhost:8000/predict"
AUDIO_SERVICE_TIMEOUT = 60
AUDIO_CONFIDENCE_THRESHOLD = 0.45
AUDIO_MIN_NOTE_DURATION = 0.05
AUDIO_DUPLICATE_WINDOW = 0.02
AUDIO_GROUPING_THRESHOLD = 0.05

GUITAR_DETECT_CONFIDENCE = 0.2

FRET_IOU_THRESHOLD = 0.9
STRING_OUTER_GAP_RATIO = 0.18
FRET_HORIZONTAL_OVERLAP = 0.2

MAX_FRETS = 25

TAB_MAX_COLS = 40

HAND_TRACKING_STEP = 2
HAND_BOX_PADDING = 0.1

FUSION_PERFECT_SCORE = 100
FUSION_UNSEEN_PENALTY = -100

EPSILON = 1e-6

# Logging configuration
LOG_LEVEL = "WARNING"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

MODEL_PATH = MODELS_DIR / "guitar_model.pt"

STANDARD_TUNING = [
    40,  # E2
    45,  # A2
    50,  # D3
    55,  # G3
    59,  # B3
    64,  # E4
]
