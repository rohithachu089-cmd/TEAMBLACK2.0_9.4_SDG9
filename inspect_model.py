import tensorflow as tf
import numpy as np
import os

model_path = "models/equipment_guard_int8.tflite"
if not os.path.exists(model_path):
    print(f"❌ MODEL NOT FOUND: {model_path}")
    exit()

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

print("--- TFLITE CORE AUDIT ---")
print(f"INPUT DTYPE: {in_det['dtype']}")
print(f"INPUT SHAPE: {in_det['shape']}")
print(f"INPUT QUANT: {in_det['quantization_parameters']}")
print(f"OUTPUT QUANT: {out_det['quantization_parameters']}")
print("-------------------------")
