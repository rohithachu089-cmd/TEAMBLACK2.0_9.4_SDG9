from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>✅ Plant Guard Server Working!</h1><p>Server time: {{ time }}</p>'

@app.route('/test')
def test():
    return {'status': 'ok', 'message': 'Flask is working!'}

if __name__ == '__main__':
    import time
    print('🚀 Starting minimal test server...')
    print('🌐 Access: http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=True)
