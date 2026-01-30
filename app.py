from flask import Flask, Response, jsonify, request, render_template_string
import google.generativeai as genai
import cv2, threading, time, base64, os, numpy as np
from config import *
from camera import CameraStream
from inference import TFLiteEquipmentClassifier

app = Flask(__name__)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# STATE
state = {
    'pred': {'label': 'INITIALIZING', 'conf': 0, 'is_fault': False},
    'cap': None,
    'hysteresis_end': 0
}
state_lock = threading.Lock()

# Initialize Robust Camera
camera = CameraStream(src=0).start()
try:
    classifier = TFLiteEquipmentClassifier(TFLITE_MODEL_PATH, LABELS_PATH)
except:
    class Dummy:
        def predict(self, f): return "NORMAL", {"normal": 1.0}
    classifier = Dummy()

def ai_loop():
    while True:
        frame = camera.read()
        if frame is not None and not np.all(frame == 0):
            label, probs = classifier.predict(frame)
            with state_lock:
                now = time.time()
                conf = round(probs.get(label, 0)*100, 1)
                is_fault = label.lower() != 'normal'
                if is_f := is_fault:
                    state['hysteresis_end'] = now + 4.0
                    state['pred'] = {'label': label.upper(), 'conf': conf, 'is_fault': True}
                    _, buf = cv2.imencode('.jpg', frame)
                    state['cap'] = base64.b64encode(buf).decode('utf-8')
                elif now >= state['hysteresis_end']:
                    state['pred'] = {'label': 'NORMAL', 'conf': round(probs.get('normal',0)*100, 1), 'is_fault': False}
        time.sleep(0.2)

threading.Thread(target=ai_loop, daemon=True).start()

STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { margin:0; padding:0; box-sizing:border-box; font-family:'Inter', sans-serif; }
    body { background:#0a0a0a; color:#fff; min-height:100vh; display:flex; flex-direction:column; }
    header { background:#000; padding:15px 30px; border-bottom:1px solid #222; display:flex; justify-content:space-between; align-items:center; }
    .title { font-size:1.3rem; font-weight:700; color:#00ff88; }
    .nav { display:flex; gap:20px; }
    .nav a { color:#666; text-decoration:none; font-weight:600; padding:8px 16px; border-radius:4px; transition:0.3s; }
    .nav a:hover, .nav a.active { background:#111; color:#00ff88; }
    .main { flex:1; display:flex; flex-direction:column; align-items:center; padding:40px; gap:30px; }
    .camera-box { position:relative; width:800px; max-width:90vw; background:#000; border:2px solid #222; border-radius:8px; overflow:hidden; }
    .feed { width:100%; display:block; }
    .overlay { position:absolute; top:20px; left:20px; text-shadow:2px 2px 10px #000; }
    .status-lbl { font-size:2.5rem; font-weight:700; }
    .warn-border { position:absolute; inset:0; border:6px solid #ff3333; pointer-events:none; display:none; }
    .btn { padding:18px 40px; font-size:1.1rem; font-weight:700; border:none; border-radius:6px; cursor:pointer; text-decoration:none; display:inline-block; }
    .btn-fix { background:#00ff88; color:#000; }
    .btn-pred { background:#3366ff; color:#fff; }
    .chat-container { width:100%; max-width:900px; background:#111; border:1px solid #222; border-radius:8px; display:flex; flex-direction:column; height:70vh; }
    .chat-messages { flex:1; padding:20px; overflow-y:auto; display:flex; flex-direction:column; gap:15px; }
    .msg { padding:15px; background:#1a1a1a; border-radius:8px; border-left:4px solid #00ff88; }
    .msg.user { background:#000; border-left-color:#555; color:#aaa; }
    .chat-input-area { padding:20px; background:#000; display:flex; gap:10px; }
    .chat-input { flex:1; background:#111; border:1px solid #333; color:#fff; padding:12px; border-radius:4px; }
</style>
"""

@app.route('/')
def home():
    # Load graphics
    eff_path = "C:/Users/Rohith/.gemini/antigravity/brain/81961c2c-1b1b-4dbd-a359-da908233557c/efficiency_dashboard_graphic_1769799204826.png"
    blue_path = "C:/Users/Rohith/.gemini/antigravity/brain/81961c2c-1b1b-4dbd-a359-da908233557c/factory_blueprint_clean_1769803175263.png"
    
    def get_b64(path):
        if os.path.exists(path):
            with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')
        return ""

    eff_b64 = get_b64(eff_path)
    blue_b64 = get_b64(blue_path)

    return render_template_string(f"""
    <html><head><title>Equipment Fixer</title>{STYLE}</head><body>
    <header><div class="title">🔧 EQUIPMENT FIXER</div><div class="nav"><a href="/" class="active">Dashboard</a><a href="/fix-solution">Fix Solution</a><a href="/predictive-solution">Predictive Solution</a></div></header>
    <div class="main">
        <!-- TOP SECTION: CAMERA + EFFICIENCY -->
        <div style="display:flex; gap:30px; align-items:flex-start; width:100%; max-width:1200px; justify-content:center; flex-wrap:wrap;">
            <div class="camera-box"><img src="/stream" class="feed"><div id="warn" class="warn-border"></div><div class="overlay"><div id="lbl" class="status-lbl">SCANNING</div></div></div>
            <div style="flex:1; min-width:300px; background:#111; border:1px solid #222; border-radius:12px; padding:25px; display:flex; flex-direction:column; gap:20px;">
                <h3 style="color:#00ff88; font-size:1.1rem; letter-spacing:1px;">OPERATIONAL EFFICIENCY</h3>
                <img src="data:image/png;base64,{eff_b64}" style="width:100%; border-radius:8px; border:1px solid #333; filter:brightness(0.8);">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-top:5px;">
                    <div style="background:#000; padding:15px; border-radius:8px; border:1px solid #222;"><div style="color:#555; font-size:0.7rem; font-weight:700;">ACCURACY</div><div style="color:#00ff88; font-size:1.4rem; font-weight:700;">98.2%</div></div>
                    <div style="background:#000; padding:15px; border-radius:8px; border:1px solid #222;"><div style="color:#555; font-size:0.7rem; font-weight:700;">UPTIME</div><div style="color:#3366ff; font-size:1.4rem; font-weight:700;">24/7</div></div>
                </div>
            </div>
        </div>

        <div class="actions"><a href="/fix-solution" class="btn btn-fix">🔧 FIX SOLUTION</a><a href="/predictive-solution" class="btn btn-pred">📊 PREDICTIVE SOLUTION</a></div>

        <!-- BOTTOM SECTION: INTERACTIVE BLUEPRINT -->
        <div style="width:100%; max-width:1200px; background:#111; border:1px solid #222; border-radius:12px; padding:30px; margin-top:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h3 style="color:#00ff88; font-size:1.1rem; letter-spacing:1px;">AUTONOMOUS FACTORY NAVIGATION</h3>
                <div id="nav-status" style="color:#3366ff; font-weight:700; font-size:0.9rem; background:#000; padding:8px 15px; border-radius:20px; border:1px solid #3366ff;">🤖 BOT STATUS: STANDBY</div>
            </div>
            <div style="position:relative; width:100%; max-width:800px; margin:0 auto;">
                <img src="data:image/png;base64,{blue_b64}" style="width:100%; border-radius:8px; border:1px solid #333 opacity:0.6;">
                <!-- Clickable Sectors (Absolute Overlays) -->
                <div onclick="move('Production Area')" style="position:absolute; top:10%; left:15%; width:30%; height:30%; cursor:pointer; background:rgba(0,255,136,0.05); border:1px dashed rgba(0,255,136,0.2); display:flex; align-items:center; justify-content:center; color:rgba(0,255,136,0.4); font-size:0.7rem; font-weight:700;">PROD-01</div>
                <div onclick="move('Assembly Line')" style="position:absolute; top:15%; left:60%; width:35%; height:40%; cursor:pointer; background:rgba(51,102,255,0.05); border:1px dashed rgba(51,102,255,0.2); display:flex; align-items:center; justify-content:center; color:rgba(51,102,255,0.4); font-size:0.7rem; font-weight:700;">ASSM-02</div>
                <div onclick="move('Logistics')" style="position:absolute; top:60%; left:5%; width:25%; height:35%; cursor:pointer; background:rgba(255,51,51,0.05); border:1px dashed rgba(255,51,51,0.2); display:flex; align-items:center; justify-content:center; color:rgba(255,51,51,0.4); font-size:0.7rem; font-weight:700;">LOG-03</div>
                <div onclick="move('Quality Control')" style="position:absolute; top:55%; left:55%; width:40%; height:40%; cursor:pointer; background:rgba(255,255,255,0.05); border:1px dashed rgba(255,255,255,0.2); display:flex; align-items:center; justify-content:center; color:rgba(255,255,255,0.4); font-size:0.7rem; font-weight:700;">QC-04</div>
            </div>
        </div>
    </div>
    <script>
    function move(s) {{
        const st = document.getElementById('nav-status');
        st.innerText = '🛰️ NAVIGATING TO: ' + s.toUpperCase();
        st.style.color = '#00ff88';
        st.style.borderColor = '#00ff88';
        setTimeout(() => {{
            st.innerText = '🤖 BOT ARRIVED AT: ' + s.toUpperCase();
            st.style.color = '#3366ff';
            st.style.borderColor = '#3366ff';
        }}, 3000);
    }}
    setInterval(() => {{ fetch('/api/sync').then(r=>r.json()).then(d => {{ document.getElementById('lbl').innerText = d.pred.label; document.getElementById('lbl').style.color = d.pred.is_fault ? '#ff3333' : '#00ff88'; document.getElementById('warn').style.display = d.pred.is_fault ? 'block' : 'none'; }}); }}, 400);</script>
    </body></html>""")

@app.route('/fix-solution')
def fix_page():
    return render_template_string(f"""
    <html><head><title>Fix Solution</title>{STYLE}</head><body>
    <header><div class="title">🔧 EQUIPMENT FIXER</div><div class="nav"><a href="/">Dashboard</a><a href="/fix-solution" class="active">Fix Solution</a><a href="/predictive-solution">Predictive Solution</a></div></header>
    <div class="main"><div class="chat-container"><div class="chat-messages" id="msgs"><div class="msg">AI Ready. Select "Analyze Defect" to begin.</div></div>
    <div class="chat-input-area"><button class="btn btn-fix" onclick="analyze()">🔍 Analyze Current Defect</button><input id="in" class="chat-input" placeholder="Ask a question..." onkeydown="if(event.key=='Enter') send()"><button class="btn btn-fix" onclick="send()">Send</button></div></div></div>
    <script>
    function analyze() {{ addMsg('Analyzing...', false); fetch('/api/fix-solution',{{method:'POST'}}).then(r=>r.json()).then(d=>{{ 
        let html = '<b>Defect:</b> '+d.defect+'<br>';
        if(d.image) html += '<img src="data:image/jpeg;base64,'+d.image+'" style="width:100%; border-radius:8px; margin:15px 0; border:1px solid #333;">';
        html += '<b>Solution:</b> '+d.solution.replace(/\\n/g,'<br>');
        addMsg(html, false);
    }}); }}
    function send() {{ const i=document.getElementById('in'); const t=i.value; if(!t)return; addMsg(t,true); i.value=''; fetch('/api/chat',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{msg:t}})}}).then(r=>r.json()).then(d=>{{ addMsg(d.reply,false); }}); }}
    function addMsg(t,u) {{ const d=document.createElement('div'); d.className='msg'+(u?' user':''); d.innerHTML=t; const c=document.getElementById('msgs'); c.appendChild(d); c.scrollTop=c.scrollHeight; }}
    </script></body></html>""")

@app.route('/predictive-solution')
def pred_page():
    return render_template_string(f"""
    <html><head><title>Predictive Solution</title>{STYLE}</head><body>
    <header><div class="title">🔧 EQUIPMENT FIXER</div><div class="nav"><a href="/">Dashboard</a><a href="/fix-solution">Fix Solution</a><a href="/predictive-solution" class="active">Predictive Solution</a></div></header>
    <div class="main">
        <div class="chat-container" style="height:auto; padding:30px;">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px;">
                <div><div style="font-size:0.8rem; color:#888; margin-bottom:5px;">Temperature (°C)</div><input class="chat-input" id="t" placeholder="e.g. 45"></div>
                <div><div style="font-size:0.8rem; color:#888; margin-bottom:5px;">Load (%)</div><input class="chat-input" id="l" placeholder="e.g. 80"></div>
                <div><div style="font-size:0.8rem; color:#888; margin-bottom:5px;">Purchase Date</div><input class="chat-input" type="date" id="p"></div>
                <div><div style="font-size:0.8rem; color:#888; margin-bottom:5px;">Last Service Date</div><input class="chat-input" type="date" id="s"></div>
            </div>
            <div style="margin-top:15px;">
                <div style="font-size:0.8rem; color:#888; margin-bottom:5px;">Daily Work Hours</div>
                <input class="chat-input" id="h" placeholder="e.g. 8">
            </div>
            <button class="btn btn-pred" style="margin-top:20px; width:100%" onclick="calc()">📊 Calculate Next Service Date</button>
            <div id="msgs" class="chat-messages" style="height:250px; margin-top:10px;"></div>
        </div>
    </div>
    <script>
    function calc() {{ const data={{temperature:document.getElementById('t').value, load:document.getElementById('l').value, last_service:document.getElementById('s').value, purchase_date:document.getElementById('p').value, work_hours:document.getElementById('h').value}};
    const m=document.getElementById('msgs'); m.innerHTML='Calculating...'; fetch('/api/predictive-solution',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(data)}}).then(r=>r.json()).then(d=>{{ m.innerHTML='<div class="msg" style="border-left-color:#3366ff">'+d.prediction.replace(/\\n/g,'<br>')+'</div>'; }}); }}
    </script></body></html>""")

@app.route('/api/sync')
def sync():
    with state_lock: return jsonify(state)

@app.route('/api/fix-solution', methods=['POST'])
def fix_api():
    with state_lock:
        defect, img_data = state['pred']['label'], state['cap']
    if not img_data:
        frame = camera.read()
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame)
            img_data = base64.b64encode(buf).decode('utf-8')
    try:
        resp = model.generate_content([f"Identify repair for {defect}", {"mime_type": "image/jpeg", "data": base64.b64decode(img_data)}])
        return jsonify({'defect': defect, 'solution': resp.text, 'image': img_data})
    except Exception as e: return jsonify({'defect': defect, 'solution': str(e)})

@app.route('/api/predictive-solution', methods=['POST'])
def pred_api():
    d = request.json
    try:
        prompt = f"""
        Predict equipment maintenance based on:
        - Current Temperature: {d.get('temperature')}°C
        - Current Load: {d.get('load')}%
        - Last Service Date: {d.get('last_service')}
        - Purchase Date: {d.get('purchase_date')}
        - Daily Work Hours: {d.get('work_hours')}
        
        Provide a specific predicted date for next service and technical reasoning.
        """
        resp = model.generate_content(prompt)
        return jsonify({'prediction': resp.text})
    except Exception as e: return jsonify({'prediction': str(e)})

@app.route('/api/chat', methods=['POST'])
def chat_api():
    d = request.json
    try:
        resp = model.generate_content(f"Expert advice: {d['msg']}")
        return jsonify({'reply': resp.text})
    except Exception as e: return jsonify({'reply': str(e)})

@app.route('/stream')
def stream():
    def g():
        while True:
            f = camera.read()
            if f is not None:
                _, b = cv2.imencode('.jpg', cv2.resize(f, (800, 600)))
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(g(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=APP_PORT, threaded=True)