# Configuration for Equipment Fixer
GEMINI_API_KEY = "AIzaSyAXwwLmR0hR6RdBqEnKFWldVGTI0YVAA5Y"

# === Model & Inference ===
MODEL_DIR = "models"
TFLITE_MODEL_PATH = f"{MODEL_DIR}/equipment_guard_int8.tflite"
LABELS_PATH = f"{MODEL_DIR}/labels.txt"

# Standard Settings
MODEL_INPUT_SIZE = (224, 224)
PREDICTION_SMOOTHING = 5
PREDICTION_THRESHOLD = 0.65
INFERENCE_FPS = 4

# === Camera ===
CAMERA_FPS = 24
CAMERA_RESOLUTION = (640, 480)

# === UI ===
APP_HOST = "0.0.0.0"
APP_PORT = 8000
DEBUG = False

# === Safety ===
# In equipment guard, we focus on notification and AI analysis
AUTO_NOTIFY_DEFAULT = True
