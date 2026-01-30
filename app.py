import os
import time
import threading
import numpy as np
import cv2
import base64
from datetime import datetime
from flask import Flask, Response, jsonify, request, render_template_string
import google.generativeai as genai

# Import configuration and modules
from config import *
from camera import CameraStream
try:
    from inference import TFLiteEquipmentClassifier as EquipmentClassifier
except ImportError:
    EquipmentClassifier = None

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==========================================
# GLOBAL STATE
# ==========================================
state = {
    'prediction': {'label': 'Initalizing System...', 'confidence': 0.0, 'probs': {}},
    'auto_mode': AUTO_NOTIFY_DEFAULT,
    'captured_image': None,
    'last_defect_time': 0,
    'status': 'System Ready'
}
state_lock = threading.Lock()

# Load Labels
LABELS = ['Normal', 'Bearing Failure', 'Overheating', 'Signal/Connection Error']
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, 'r') as f:
            LABELS = [l.strip() for l in f.readlines() if l.strip()]
    except:
        pass

# ==========================================
# INITIALIZATION
# ==========================================
camera = CameraStream(src=0, resolution=CAMERA_RESOLUTION, fps=CAMERA_FPS).start()
classifier = None

if EquipmentClassifier and os.path.exists(TFLITE_MODEL_PATH):
    try:
        classifier = EquipmentClassifier(
            model_path=TFLITE_MODEL_PATH,
            labels=LABELS,
            input_size=MODEL_INPUT_SIZE
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# ==========================================
# BACKGROUND THREADS
# ==========================================
def inference_loop():
    print("⚙️ Inference engine started")
    global state
    while True:
        if classifier and camera.frame is not None:
            try:
                label, probs = classifier.predict(camera.frame, threshold=PREDICTION_THRESHOLD)
                with state_lock:
                    conf = probs.get(label, 0.0) * 100
                    state['prediction'] = {
                        'label': label if label != "unknown" else "Scanning...",
                        'confidence': round(conf, 1),
                        'probs': probs
                    }
                    
                    # Defect Capture Logic
                    if label not in ['Normal', 'Scanning...', 'unknown'] and conf > (PREDICTION_THRESHOLD * 100):
                        current_time = time.time()
                        if current_time - state['last_defect_time'] > 10: # 10s cooldown
                            _, buffer = cv2.imencode('.jpg', camera.frame)
                            state['captured_image'] = base64.b64encode(buffer).decode('utf-8')
                            state['last_defect_time'] = current_time
                            
            except Exception as e:
                print(f"⚠️ Inference error: {e}")
        time.sleep(1.0 / INFERENCE_FPS)

if classifier:
    threading.Thread(target=inference_loop, daemon=True).start()

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_frames():
    while True:
        frame = camera.read()
        if frame is None: continue
            
        with state_lock:
            pred = state['prediction']
            label = pred['label']
            conf = pred['confidence']
        
        # B&W HUD Overlay
        color = (255, 255, 255) # White default
        if label not in ['Normal', 'Scanning...']:
            color = (0, 0, 255) # Red highlight for defect
            # Draw highlight box for defect
            h, w, _ = frame.shape
            cv2.rectangle(frame, (50, 50), (w-50, h-50), color, 2)
            cv2.putText(frame, "DEFECT DETECTED", (w//2-100, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # HUD Bars
        cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"EQUIPMENT GUARD | {label.upper()}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1.0 / 30)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with state_lock:
        return jsonify(state)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    is_repair = data.get('is_repair', False)
    
    try:
        if is_repair:
            with state_lock:
                img_data = state['captured_image']
                label = state['prediction']['label']
            
            if not img_data:
                return jsonify({'reply': "No defect image captured yet. Please point the camera at the equipment."})
            
            # Send image to Gemini
            content = [
                f"The system detected a potential defect: {label}. Based on this image, please provide a professional technical solution and maintenance steps.",
                {'mime_type': 'image/jpeg', 'data': base64.b64decode(img_data)}
            ]
            response = model.generate_content(content)
            return jsonify({'reply': response.text, 'image': img_data})
        else:
            # Predictive maintenance chat
            response = model.generate_content(f"User is asking about industrial equipment maintenance. Input: {user_msg}. Provide a concise, professional technical recommendation based on temperature, vibration, and general equipment health.")
            return jsonify({'reply': response.text})
    except Exception as e:
        return jsonify({'reply': f"Error connecting to AI: {str(e)}"})

# ==========================================
# B&W CLASSIC UI
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EQUIPMENT GUARD AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #000000;
            --surface: #111111;
            --border: #333333;
            --accent: #ffffff;
            --danger: #ff0000;
            --font-main: 'Inter', sans-serif;
            --font-mono: 'Roboto Mono', monospace;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            background: var(--bg); 
            color: var(--accent); 
            font-family: var(--font-main);
            overflow-x: hidden;
        }
        .header {
            padding: 30px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo { font-size: 1.5rem; font-weight: 700; letter-spacing: 4px; }
        
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            min-height: calc(100vh - 100px);
        }
        
        .monitor-section {
            padding: 20px;
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-container {
            width: 100%;
            max-width: 800px;
            border: 1px solid var(--border);
            position: relative;
        }
        .video-container img { width: 100%; display: block; }
        
        .prediction-banner {
            width: 100%;
            max-width: 800px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid var(--border);
            background: var(--surface);
        }
        .label-text { font-family: var(--font-mono); font-size: 2rem; }
        .conf-text { color: #666; font-size: 0.9rem; }
        
        .chat-section {
            padding: 20px;
            display: flex;
            flex-direction: column;
            background: #050505;
        }
        .chat-header {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 20px;
            color: #444;
        }
        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            margin-bottom: 20px;
            padding-right: 10px;
        }
        .message {
            margin-bottom: 15px;
            font-size: 0.9rem;
            line-height: 1.5;
            padding: 10px;
            border-left: 2px solid var(--border);
        }
        .message.user { border-left-color: #666; color: #aaa; }
        .message.ai { border-left-color: #fff; }
        
        .chat-input-area {
            display: flex;
            gap: 10px;
        }
        input {
            flex-grow: 1;
            background: #000;
            border: 1px solid var(--border);
            color: #fff;
            padding: 10px;
            font-family: var(--font-main);
        }
        button {
            background: #fff;
            color: #000;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.7rem;
            transition: all 0.2s;
        }
        button:hover { background: #ccc; }
        button.danger { background: var(--danger); color: #fff; }
        
        .fix-button {
            margin-top: 15px;
            width: 100%;
            display: none;
        }
        
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: var(--border); }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">EQUIPMENT GUARD AI</div>
        <div id="status-tag" style="font-size: 0.7rem; color: #444;">ENGINE ONLINE</div>
    </div>
    
    <div class="main-grid">
        <div class="monitor-section">
            <div class="video-container">
                <img src="/video_feed" alt="Live Stream">
            </div>
            
            <div class="prediction-banner">
                <div class="conf-text">CURRENT STATUS</div>
                <div class="label-text" id="label-val">CONNECTING...</div>
                <div class="conf-text" id="conf-val">Inference: 0%</div>
                
                <button id="fix-btn" class="fix-button danger" onclick="requestFix()">🔧 FIX DETECTED DAMAGE</button>
            </div>
        </div>
        
        <div class="chat-section">
            <div class="chat-header">Maintenance Intelligence Core</div>
            <div class="chat-messages" id="chat-msgs">
                <div class="message ai">Welcome. I am the Equipment Maintenance Assistant. You can ask me about temperature, vibration, or general maintenance, or use the 'Fix' button when a defect is detected.</div>
            </div>
            <div class="chat-input-area">
                <input type="text" id="chat-input" placeholder="Type data (temp, vib) or query...">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const p = data.prediction;
                    document.getElementById('label-val').textContent = p.label.toUpperCase();
                    document.getElementById('conf-val').textContent = 'Confidence: ' + p.confidence + '%';
                    
                    const fixBtn = document.getElementById('fix-btn');
                    if (p.label !== 'Normal' && p.label !== 'Scanning...' && p.confidence > 60) {
                        fixBtn.style.display = 'block';
                        document.getElementById('label-val').style.color = '#ff0000';
                    } else {
                        fixBtn.style.display = 'none';
                        document.getElementById('label-val').style.color = '#ffffff';
                    }
                });
        }

        function addMessage(text, role) {
            const msgs = document.getElementById('chat-msgs');
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.innerHTML = text.replace(/\\n/g, '<br>');
            msgs.appendChild(div);
            msgs.scrollTop = msgs.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('chat-input');
            const msg = input.value;
            if (!msg) return;
            
            addMessage(msg, 'user');
            input.value = '';
            
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: msg})
            })
            .then(r => r.json())
            .then(data => addMessage(data.reply, 'ai'));
        }

        function requestFix() {
            addMessage("Requesting repair solution for detected defect...", 'user');
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({is_repair: true})
            })
            .then(r => r.json())
            .then(data => {
                if (data.image) {
                    const imgHtml = `<img src="data:image/jpeg;base64,${data.image}" style="width:100%; border:1px solid #333; margin:10px 0;">`;
                    addMessage(imgHtml + data.reply, 'ai');
                } else {
                    addMessage(data.reply, 'ai');
                }
            });
        }

        setInterval(updateStatus, 1000);
        
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print(f"🚀 Equipment Guard AI running on port {APP_PORT}")
    try:
        app.run(host=APP_HOST, port=APP_PORT, debug=False, threaded=True)
    finally:
        if camera:
            camera.stop()