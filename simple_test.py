from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World! Plant Guard Server is Running!'

@app.route('/test')
def test():
    return 'TEST OK - Flask is working!'

if __name__ == '__main__':
    print('=' * 50)
    print('🚀 SIMPLE PLANT GUARD TEST')
    print('=' * 50)
    print('Starting server on http://localhost:5000')
    print('=' * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
