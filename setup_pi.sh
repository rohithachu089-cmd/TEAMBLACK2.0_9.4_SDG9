#!/bin/bash

# Equipment Guard AI - Raspberry Pi Setup Script
# Run this on your Pi: bash setup_pi.sh

echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-opencv libatlas-base-dev libjpeg-dev libopenjp2-7

echo "🐍 Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "pip upgrading..."
pip install --upgrade pip

echo "🤖 Installing TensorFlow Lite Runtime..."
# For Raspberry Pi 64-bit (recommended)
pip install tflite-runtime==2.14.0

echo "📚 Installing Project Requirements..."
pip install flask==2.3.3 numpy==1.26.4 requests==2.31.0 pillow==10.2.0 scikit-learn==1.3.2 google-generativeai==0.8.3

echo "✅ Setup Complete!"
echo "To run the app: source .venv/bin/activate && python app.py"
