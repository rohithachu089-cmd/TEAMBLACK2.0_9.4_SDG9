# Configuration for Equipment Guard AI

# === Gemini API (for AI Chatbot) ===
GEMINI_API_KEY = "AIzaSyAegnoZWj22vanWah48rnvKDLcDuAxTMiM"

# === Model & Inference ===
MODEL_DIR = "models"
TFLITE_MODEL_PATH = f"{MODEL_DIR}/equipment_guard_int8.tflite"
LABELS_PATH = f"{MODEL_DIR}/labels.txt"

# MobileNetV2 default input size
MODEL_INPUT_SIZE = (224, 224)  # (H, W)
PREDICTION_SMOOTHING = 5       # Smooth out jitter
PREDICTION_THRESHOLD = 0.65    # Threshold for equipment defects
INFERENCE_FPS = 4              # Run N inferences per second

# === Camera ===
CAMERA_FPS = 30                # Higher FPS for smoother stream
CAMERA_RESOLUTION = (640, 480)

# === UI ===
APP_HOST = "0.0.0.0"
APP_PORT = 8000
DEBUG = False

# === Safety ===
# In equipment guard, we focus on notification and AI analysis
AUTO_NOTIFY_DEFAULT = True
