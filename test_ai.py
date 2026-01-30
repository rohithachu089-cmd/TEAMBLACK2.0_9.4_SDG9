# Final AI Core Verification (v7.0)
import cv2
import time
import os
import numpy as np
from config import *
from inference import TFLiteEquipmentClassifier

def run_final_audit():
    print("🔬 INITIALIZING FINAL CORE AUDIT (v7.0)...")
    classifier = TFLiteEquipmentClassifier(TFLITE_MODEL_PATH, LABELS_PATH)
    
    cap = cv2.VideoCapture(0)
    time.sleep(2) # Warmup
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ CAMERA_ERROR: Could not grab frame.")
        return

    # Save the exactly seen frame for user inspection
    cv2.imwrite("last_inspection_frame.jpg", frame)
    print("📸 FRAME CAPTURED and saved to 'last_inspection_frame.jpg'")

    print("\n--- INFERENCE AUDIT ---")
    label, probs = classifier.predict(frame, threshold=0.1)
    
    print(f"DECISION: {label.upper()}")
    for l, p in probs.items():
        print(f"-> {l}: {p:.4f}")
    print("-"*20)
    
    if label.lower() == 'normal' and probs.get('normal', 0) > 0.99:
        print("⚠️ WARNING: Absolute 'Normal' bias detected. Check if the camera is covered or lighting is too dark.")
    else:
        print("✅ SUCCESS: Diversified probabilities detected. The core is receptive to features.")

if __name__ == "__main__":
    run_final_audit()
