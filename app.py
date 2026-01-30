from flask import Flask, Response, jsonify, request, render_template_string
import google.generativeai as genai
import cv2, threading, time, base64, os, numpy as np
from datetime import datetime, timedelta
from config import *
from camera import CameraStream
from inference import TFLiteEquipmentClassifier

# v15.3 NEW API KEY & EQUIPMENT FIXER
VERSION = "15.3 - EQUIPMENT_FIXER"
PORT = 8888

app = Flask(__name__)
genai.configure(api_key=GEMINI_API_KEY)
# Use gemini-1.5-pro for vision
model = genai.GenerativeModel('gemini-1.5-flash')

# STATE
state = {
    'pred': {'label': 'INITIALIZING', 'conf': 0, 'is_fault': False},
    'cap': None,
    'full_map': {},
    'hysteresis_end': 0
}
state_lock = threading.Lock()

camera = CameraStream(src=0).start()
classifier = TFLiteEquipmentClassifier(TFLITE_MODEL_PATH, LABELS_PATH)

def ai_loop():
    print(f"🚀 EQUIPMENT FIXER v{VERSION} | PORT {PORT}")
    while True:
        frame = camera.read()
        if frame is not None and not np.all(frame == 0):
            label, probs = classifier.predict(frame)
            
            with state_lock:
                now = time.time()
                conf = round(probs.get(label, 0)*100, 1)
                is_fault = label.lower() != 'normal'
                
                if is_fault:
                    state['hysteresis_end'] = now + 4.0
                    state['pred'] = {'label': label.upper(), 'conf': conf, 'is_fault': True}
                    _, buf = cv2.imencode('.jpg', frame)
                    state['cap'] = base64.b64encode(buf).decode('utf-8')
                elif now < state['hysteresis_end']:
                    pass
                else:
                    state['pred'] = {'label': 'NORMAL', 'conf': round(probs.get('normal',0)*100, 1), 'is_fault': False}
                
                state['full_map'] = {k.upper(): round(v*100, 1) for k,v in probs.items()}
        time.sleep(0.2)

threading.Thread(target=ai_loop, daemon=True).start()

@app.after_request
def nocache(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return r

STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { margin:0; padding:0; box-sizing:border-box; font-family:'Inter', sans-serif; }
    body { background:#0a0a0a; color:#fff; min-height:100vh; display:flex; flex-direction:column; }
    
    header { background:#000; padding:15px 30px; border-bottom:1px solid #222; display:flex; justify-content:space-between; align-items:center; }
    .title { font-size:1.3rem; font-weight:700; color:#00ff88; }
    .nav { display:flex; gap:20px; }
    .nav a { color:#666; text-decoration:none; font-weight:600; padding:8px 16px; border-radius:4px; transition:0.3s; }
    .nav a:hover { background:#111; color:#00ff88; }
    .nav a.active { background:#00ff88; color:#000; }
    
    .main { flex:1; display:flex; flex-direction:column; align-items:center; padding:40px; gap:30px; }
    
    .camera-box { position:relative; width:800px; max-width:90vw; background:#000; border:2px solid #222; border-radius:8px; overflow:hidden; }
    .feed { width:100%; display:block; }
    .overlay { position:absolute; top:20px; left:20px; }
    .status-lbl { font-size:2.5rem; font-weight:700; text-shadow:0 2px 8px rgba(0,0,0,0.9); }
    .conf-lbl { font-size:1rem; color:#aaa; margin-top:5px; }
    .warn-border { position:absolute; inset:0; border:6px solid #ff3333; pointer-events:none; display:none; }
    
    .actions { display:flex; gap:20px; }
    .btn { padding:18px 40px; font-size:1.1rem; font-weight:700; border:none; border-radius:6px; cursor:pointer; transition:0.3s; text-transform:uppercase; text-decoration:none; display:inline-block; }
    .btn-fix { background:#00ff88; color:#000; }
    .btn-fix:hover { background:#00dd77; transform:translateY(-2px); }
    .btn-pred { background:#3366ff; color:#fff; }
    .btn-pred:hover { background:#2255ee; transform:translateY(-2px); }
    
    /* CHAT PAGE STYLES */
    .chat-container { flex:1; width:100%; max-width:1200px; display:flex; flex-direction:column; background:#111; border-radius:8px; border:1px solid #222; overflow:hidden; }
    .chat-header { background:#000; padding:20px; border-bottom:1px solid #222; }
    .chat-title { font-size:1.5rem; font-weight:700; }
    .chat-title.fix { color:#00ff88; }
    .chat-title.pred { color:#3366ff; }
    
    .chat-messages { flex:1; padding:30px; overflow-y:auto; display:flex; flex-direction:column; gap:20px; min-height:400px; }
    .msg { padding:20px; background:#1a1a1a; border-radius:8px; line-height:1.6; font-size:1rem; border-left:4px solid #00ff88; }
    .msg.pred { border-left-color:#3366ff; }
    .msg.user { background:#0a0a0a; border-left-color:#555; color:#aaa; }
    
    .chat-options { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; padding:20px; background:#000; border-top:1px solid #222; }
    .opt-btn { padding:15px; background:#111; border:1px solid #333; color:#aaa; font-weight:600; border-radius:6px; cursor:pointer; transition:0.3s; text-align:center; }
    .opt-btn:hover { border-color:#00ff88; color:#fff; background:#1a1a1a; }
    .opt-btn.pred:hover { border-color:#3366ff; }
    
    .chat-input-area { padding:20px; background:#000; display:flex; gap:12px; border-top:1px solid #222; }
    .chat-input { flex:1; background:#111; border:1px solid #333; color:#fff; padding:15px; border-radius:6px; font-size:1rem; }
    .chat-input:focus { outline:none; border-color:#00ff88; }
    .send-btn { background:#00ff88; color:#000; border:none; padding:0 30px; border-radius:6px; font-weight:700; cursor:pointer; font-size:1rem; }
    .send-btn.pred { background:#3366ff; color:#fff; }
    
    .pred-form { padding:20px; background:#000; border-top:1px solid #222; }
    .form-grid { display:grid; grid-template-columns:1fr 1fr; gap:15px; }
    .form-row { margin-bottom:15px; }
    .form-label { font-size:0.85rem; color:#888; margin-bottom:6px; display:block; font-weight:600; }
    .form-input { width:100%; background:#111; border:1px solid #333; color:#fff; padding:12px; border-radius:6px; font-size:0.95rem; }
    .form-input:focus { outline:none; border-color:#3366ff; }
    .calc-btn { width:100%; background:#3366ff; color:#fff; border:none; padding:15px; border-radius:6px; font-weight:700; cursor:pointer; margin-top:15px; font-size:1rem; }
    .calc-btn:hover { background:#2255ee; }
    
    .loading { text-align:center; color:#666; padding:30px; font-size:1.1rem; }
</style>
"""

@app.route('/')
def home():
    return render_template_string(f"""
    <html><head><title>Equipment Fixer</title>{STYLE}</head><body>
    <header>
        <div class="title">🔧 EQUIPMENT FIXER</div>
        <div class="nav">
            <a href="/" class="active">Dashboard</a>
            <a href="/fix-solution">Fix Solution</a>
            <a href="/predictive-solution">Predictive Solution</a>
        </div>
    </header>
    
    <div class="main">
        <div class="camera-box">
            <img src="/stream" class="feed">
            <div id="warn" class="warn-border"></div>
            <div class="overlay">
                <div id="lbl" class="status-lbl">SCANNING</div>
                <div id="conf" class="conf-lbl">Initializing...</div>
            </div>
        </div>
        
        <div class="actions">
            <a href="/fix-solution" class="btn btn-fix">🔧 FIX SOLUTION</a>
            <a href="/predictive-solution" class="btn btn-pred">📊 PREDICTIVE SOLUTION</a>
        </div>
    </div>
    
    <script>
        setInterval(() => {{
            fetch('/api/sync').then(r=>r.json()).then(d => {{
                const l = d.pred.label;
                const f = d.pred.is_fault;
                document.getElementById('lbl').innerText = l;
                document.getElementById('conf').innerText = d.pred.conf + '% Confidence';
                
                if(f) {{
                    document.getElementById('lbl').style.color = '#ff3333';
                    document.getElementById('warn').style.display = 'block';
                }} else {{
                    document.getElementById('lbl').style.color = '#00ff88';
                    document.getElementById('warn').style.display = 'none';
                }}
            }});
        }}, 400);
    </script>
    </body></html>
    """)

@app.route('/fix-solution')
def fix_solution_page():
    return render_template_string(f"""
    <html><head><title>Equipment Fixer - Fix Solution</title>{STYLE}</head><body>
    <header>
        <div class="title">🔧 EQUIPMENT FIXER</div>
        <div class="nav">
            <a href="/">Dashboard</a>
            <a href="/fix-solution" class="active">Fix Solution</a>
            <a href="/predictive-solution">Predictive Solution</a>
        </div>
    </header>
    
    <div class="main">
        <div class="chat-container">
            <div class="chat-header">
                <div class="chat-title fix">🔧 Fix Solution Chat</div>
            </div>
            <div id="messages" class="chat-messages">
                <div class="msg">Fix Solution AI ready. Click options below or type your question.</div>
            </div>
            <div class="chat-options">
                <div class="opt-btn" onclick="analyze()">🔍 Analyze Current Defect</div>
                <div class="opt-btn" onclick="quick('Identify the problem')">Identify Problem</div>
                <div class="opt-btn" onclick="quick('Show repair steps')">Repair Steps</div>
                <div class="opt-btn" onclick="quick('Safety precautions')">Safety Info</div>
                <div class="opt-btn" onclick="quick('Required tools and parts')">Tools Needed</div>
                <div class="opt-btn" onclick="quick('Estimated repair time')">Time Estimate</div>
            </div>
            <div class="chat-input-area">
                <input id="input" class="chat-input" placeholder="Ask about the defect..." onkeydown="if(event.key=='Enter') send()">
                <button class="send-btn" onclick="send()">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        function analyze() {{
            addMsg('Analyzing current defect...', false);
            fetch('/api/fix-solution', {{method:'POST'}})
                .then(r=>r.json())
                .then(d => {{
                    addMsg('<b>Defect:</b> ' + d.defect + '<br><br><b>Solution:</b><br>' + d.solution.replace(/\\n/g, '<br>'), false);
                }})
                .catch(e => {{ addMsg('Error: ' + e, false); }});
        }}
        
        function quick(msg) {{
            addMsg(msg, true);
            fetch('/api/chat', {{
                method:'POST',
                headers:{{'Content-Type':'application/json'}},
                body:JSON.stringify({{type:'fix', msg:msg}})
            }})
            .then(r=>r.json())
            .then(d => {{ addMsg(d.reply, false); }})
            .catch(e => {{ addMsg('Error: ' + e, false); }});
        }}
        
        function send() {{
            const i = document.getElementById('input');
            const txt = i.value; if(!txt) return;
            addMsg(txt, true); i.value = '';
            
            fetch('/api/chat', {{
                method:'POST',
                headers:{{'Content-Type':'application/json'}},
                body:JSON.stringify({{type:'fix', msg:txt}})
            }})
            .then(r=>r.json())
            .then(d => {{ addMsg(d.reply, false); }})
            .catch(e => {{ addMsg('Error: ' + e, false); }});
        }}
        
        function addMsg(txt, isUser) {{
            const d = document.createElement('div');
            d.className = 'msg' + (isUser ? ' user' : '');
            d.innerHTML = txt;
            const c = document.getElementById('messages');
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }}
    </script>
    </body></html>
    """)

@app.route('/predictive-solution')
def predictive_solution_page():
    return render_template_string(f"""
    <html><head><title>Equipment Fixer - Predictive Solution</title>{STYLE}</head><body>
    <header>
        <div class="title">🔧 EQUIPMENT FIXER</div>
        <div class="nav">
            <a href="/">Dashboard</a>
            <a href="/fix-solution">Fix Solution</a>
            <a href="/predictive-solution" class="active">Predictive Solution</a>
        </div>
    </header>
    
    <div class="main">
        <div class="chat-container">
            <div class="chat-header">
                <div class="chat-title pred">📊 Predictive Maintenance Chat</div>
            </div>
            <div id="messages" class="chat-messages">
                <div class="msg pred">Predictive Maintenance AI ready. Fill the form below to calculate next service date.</div>
            </div>
            <div class="pred-form">
                <div class="form-grid">
                    <div class="form-row">
                        <label class="form-label">Temperature (°C)</label>
                        <input type="number" class="form-input" id="temp" placeholder="45">
                    </div>
                    <div class="form-row">
                        <label class="form-label">Load (%)</label>
                        <input type="number" class="form-input" id="load" placeholder="75">
                    </div>
                    <div class="form-row">
                        <label class="form-label">Last Service Date</label>
                        <input type="date" class="form-input" id="last-service">
                    </div>
                    <div class="form-row">
                        <label class="form-label">Purchase Date</label>
                        <input type="date" class="form-input" id="purchase-date">
                    </div>
                </div>
                <div class="form-row">
                    <label class="form-label">Daily Work Hours</label>
                    <input type="number" class="form-input" id="work-hours" placeholder="8">
                </div>
                <button class="calc-btn" onclick="calculate()">📊 Calculate Next Service Date</button>
            </div>
            <div class="chat-input-area">
                <input id="input" class="chat-input" placeholder="Ask about maintenance..." onkeydown="if(event.key=='Enter') send()">
                <button class="send-btn pred" onclick="send()">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        function calculate() {{
            const data = {{
                temperature: document.getElementById('temp').value,
                load: document.getElementById('load').value,
                last_service: document.getElementById('last-service').value,
                purchase_date: document.getElementById('purchase-date').value,
                work_hours: document.getElementById('work-hours').value
            }};
            
            addMsg('Calculating next service date...', false);
            
            fetch('/api/predictive-solution', {{
                method:'POST',
                headers:{{'Content-Type':'application/json'}},
                body:JSON.stringify(data)
            }})
            .then(r=>r.json())
            .then(d => {{ addMsg(d.prediction.replace(/\\n/g, '<br>'), false); }})
            .catch(e => {{ addMsg('Error: ' + e, false); }});
        }}
        
        function send() {{
            const i = document.getElementById('input');
            const txt = i.value; if(!txt) return;
            addMsg(txt, true); i.value = '';
            
            fetch('/api/chat', {{
                method:'POST',
                headers:{{'Content-Type':'application/json'}},
                body:JSON.stringify({{type:'pred', msg:txt}})
            }})
            .then(r=>r.json())
            .then(d => {{ addMsg(d.reply, false); }})
            .catch(e => {{ addMsg('Error: ' + e, false); }});
        }}
        
        function addMsg(txt, isUser) {{
            const d = document.createElement('div');
            d.className = 'msg pred' + (isUser ? ' user' : '');
            d.innerHTML = txt;
            const c = document.getElementById('messages');
            c.appendChild(d);
            c.scrollTop = c.scrollHeight;
        }}
    </script>
    </body></html>
    """)

@app.route('/api/sync')
def sync():
    with state_lock:
        return jsonify(state)

@app.route('/api/fix-solution', methods=['POST'])
def fix_solution():
    with state_lock:
        defect = state['pred']['label']
        img_data = state['cap']
    
    if not img_data:
        frame = camera.read()
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame)
            img_data = base64.b64encode(buf).decode('utf-8')
            defect = "CURRENT_VIEW"
    
    try:
        prompt = f"""You are an industrial equipment repair expert analyzing {defect}.

Provide:
1. Problem identification
2. Step-by-step repair solution
3. Safety precautions
4. Required tools and parts

Be concise and practical."""

        parts = [prompt, {"mime_type": "image/jpeg", "data": base64.b64decode(img_data)}]
        response = model.generate_content(parts)
        
        return jsonify({
            'defect': defect,
            'solution': response.text
        })
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        return jsonify({'defect': defect, 'solution': f'Gemini API Error: {str(e)}'})

@app.route('/api/predictive-solution', methods=['POST'])
def predictive_solution():
    data = request.json
    
    try:
        prompt = f"""You are a predictive maintenance specialist.

Equipment Data:
- Temperature: {data.get('temperature', 'N/A')}°C
- Load: {data.get('load', 'N/A')}%
- Last Service: {data.get('last_service', 'N/A')}
- Purchase Date: {data.get('purchase_date', 'N/A')}
- Daily Hours: {data.get('work_hours', 'N/A')}

Provide:
1. Recommended next service date
2. Reasoning for this timeline
3. Equipment health assessment
4. Preventive maintenance tips

Be specific with dates and practical advice."""

        response = model.generate_content(prompt)
        return jsonify({'prediction': response.text})
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        return jsonify({'prediction': f'Gemini API Error: {str(e)}'})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    chat_type = data.get('type')
    msg = data.get('msg')
    
    with state_lock:
        defect = state['pred']['label']
        img_data = state['cap']
    
    if not img_data:
        frame = camera.read()
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame)
            img_data = base64.b64encode(buf).decode('utf-8')
    
    try:
        if chat_type == 'fix':
            prompt = f"You are an equipment repair expert. Current detection: {defect}. User question: {msg}. Provide a brief, focused answer."
        else:
            prompt = f"You are a maintenance planning advisor. User question: {msg}. Provide practical maintenance advice."
        
        parts = [prompt, {"mime_type": "image/jpeg", "data": base64.b64decode(img_data)}]
        response = model.generate_content(parts)
        return jsonify({'reply': response.text})
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        return jsonify({'reply': f'Gemini API Error: {str(e)}'})

@app.route('/stream')
def stream():
    def gen():
        while True:
            f = camera.read()
            if f is not None:
                f = cv2.resize(f, (800, 600))
                _, b = cv2.imencode('.jpg', f)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, threaded=True)