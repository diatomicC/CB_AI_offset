#!/bin/bash

# Simple Docker Windows Build Script

echo "🐳 Docker Windows Build (Simple Version)"
echo "=================================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    exit 1
fi

# Clean previous build
rm -rf build dist

# Build Windows with Docker
echo "🚀 Building Windows executable..."
docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    python:3.9-slim \
    bash -c "
        pip install pyinstaller PySide6 opencv-python numpy pandas openpyxl reportlab &&
        pyinstaller --onefile --windowed --icon=app_icon.ico --name=CMYK_Analyzer run_gui.py
    "

if [ $? -eq 0 ]; then
    echo "✅ Windows build completed!"
    ls -la dist/
else
    echo "❌ Windows build failed!"
fi
