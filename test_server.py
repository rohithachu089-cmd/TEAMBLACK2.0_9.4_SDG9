from flask import Flask
import time
import socket  # FIXED: This import was missing!

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <html>
    <head><title>Test Server</title></head>
    <body style="background: black; color: lime; font-family: monospace; padding: 50px; text-align: center;">
        <h1 style="font-size: 3em;">✅ Plant Guard Server</h1>
        <p style="font-size: 1.5em;">Server Time: ''' + time.strftime('%Y-%m-%d %H:%M:%S') + '''</p>
        <hr style="border: 1px solid lime; width: 50%;">
        <h2>Server Status: <span style="color: lime;">RUNNING</span></h2>
        <p>Port: 5000 | Host: localhost</p>
        <p><a href="/test" style="color: lime; text-decoration: none;">🔗 Test API</a></p>
        <hr style="border: 1px solid lime; width: 50%;">
        <h3 style="color: #00ff00;">🎉 SUCCESS! Flask is working!</h3>
    </body>
    </html>
    '''

@app.route('/test')
def test():
    return {
        'status': 'success',
        'message': 'Flask is working perfectly!',
        'timestamp': time.time(),
        'server_time': time.strftime('%H:%M:%S'),
        'python_version': sys.version
    }

if __name__ == '__main__':
    print('=' * 60)
    print('🚀 PLANT GUARD TEST SERVER - FIXED VERSION')
    print('=' * 60)
    
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
        print(f'🌐 Local:     http://localhost:5000')
        print(f'🌐 Network:   http://{local_ip}:5000')
        print('=' * 60)
        print('Starting server... (Press Ctrl+C to stop)')
        print('=' * 60)
        
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    except Exception as e:
        print(f'❌ Server error: {e}')
        import traceback
        traceback.print_exc()
