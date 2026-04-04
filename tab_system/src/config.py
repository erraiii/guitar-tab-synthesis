from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MAX_HANDS = 2


VIDEO_PATH = "C:/Users/aivan/Downloads/blue_test_short.mp4"


STANDARD_TUNING = [
    40,  # E2
    45,  # A2
    50,  # D3
    55,  # G3
    59,  # B3
    64,  # E4
]

MODEL_PATH = MODELS_DIR / "guitar_model.pt"
