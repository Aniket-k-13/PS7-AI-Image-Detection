#!/bin/bash

echo "========================================"
echo " PS7 - AI Generated Image Detector"
echo " Neural Nexus Hackathon 2026"
echo "========================================"
echo ""

echo "[1/2] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "[2/2] Launching Streamlit app..."
echo "App will open at http://localhost:8501"
echo ""

streamlit run app.py
