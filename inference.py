import numpy as np
import cv2
import os
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

class TFLiteEquipmentClassifier:
    def __init__(self, model_path, labels_path):
        # Load Labels
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.labels = ['bearing_failure', 'connect_disconnection', 'normal', 'overheating']
            
        # Init Interpreter
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.in_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]
        
        self.h, self.w = self.in_det['shape'][1], self.in_det['shape'][2]
        self.dtype = self.in_det['dtype']
        
        # Quantization Params
        q = self.in_det.get('quantization_parameters', {})
        self.in_scale = q.get('scales', [0.0])[0] if q.get('scales') else 0.0
        self.in_zp = q.get('zero_points', [0])[0] if q.get('zero_points') else 0
        
        print(f"✅ [CORE] v14.0 RECONSTRUCTED | {self.dtype} | Scale:{self.in_scale} ZP:{self.in_zp}")

    def predict(self, frame):
        if frame is None: return "error", {}
            
        # 1. Strict Resizing
        img = cv2.resize(frame, (224, 224)) # Force 224x224
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Input Preparation
        if self.dtype == np.uint8:
            # Standard Quantized Input: 0-255
            input_data = np.expand_dims(img.astype(np.uint8), axis=0)
            # If Model defines Scale/ZP, apply reverse mapping? 
            # Usually for TFLite Image models, you pass the raw uint8 image matching typical training
            # unless the metadata explicitly demands (val/scale)+zp.
            # v14 Strategy: Direct Pass for maximum compatibility with standard TFLite export
            # If this fails, we revert to manual quantization.
            # Let's try the safest path: Manual Quantization if scale exists
            if self.in_scale > 0:
                 norm = img.astype(np.float32) / 255.0
                 quant = (norm / self.in_scale) + self.in_zp
                 input_data = np.expand_dims(np.clip(quant, 0, 255).astype(np.uint8), axis=0)
        elif self.dtype == np.float32:
             # Standard Float Input: 0-1
             input_data = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
        else:
             input_data = np.expand_dims(img, axis=0) # Fallback
        
        # 3. Invoke
        self.interpreter.set_tensor(self.in_det['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.out_det['index'])[0]
        
        # 4. Process Output
        # Dequantize if needed
        oq = self.out_det.get('quantization_parameters', {})
        if oq.get('scales'):
            output = (output.astype(np.float32) - oq['zero_points'][0]) * oq['scales'][0]
            
        # Softmax
        output = output.astype(np.float32)
        output = output - np.max(output)
        exps = np.exp(output)
        probs_raw = exps / np.sum(exps)
        probs = {self.labels[i]: float(probs_raw[i]) for i in range(len(self.labels))}

        # 5. SMART SENSITIVITY LOGIC (v14)
        # Find highest defect
        best_defect = "normal"
        defect_conf = 0
        normal_conf = probs.get("normal", 0)
        
        for l, p in probs.items():
            if l.lower() != 'normal' and p > defect_conf:
                defect_conf = p
                best_defect = l
        
        # Adaptive Thresholds
        # 1. Defect is dominant (>40%)             -> FAULT
        # 2. Defect is significant (>25%)          -> FAULT
        # 3. Defect is visible (>15%) AND Normal is weak (<60%) -> FAULT
        if defect_conf > 0.40:
             final_label = best_defect
             final_conf = defect_conf
        elif defect_conf > 0.25:
             final_label = best_defect
             final_conf = defect_conf
        elif defect_conf > 0.15 and normal_conf < 0.60:
             final_label = best_defect
             final_conf = defect_conf
        else:
             final_label = "normal"
             final_conf = normal_conf

        print(f"🧠 [v14] {final_label.upper()} ({final_conf:.2f}) | N:{normal_conf:.2f} D:{defect_conf:.2f}")
        return final_label, probs
