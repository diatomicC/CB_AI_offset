#!/bin/bash

# CMYK Analyzer macOS Build Script

echo "🎯 CMYK Analyzer macOS Executable Builder"
echo "=================================================="

# Check Python virtual environment
if [ -d "venv" ]; then
    echo "✅ Virtual environment found. Activating..."
    source venv/bin/activate
else
    echo "⚠️ Virtual environment not found. Using system Python."
fi

# Check PyInstaller installation
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "📦 Installing PyInstaller..."
    pip install pyinstaller
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install PyInstaller."
        exit 1
    fi
    echo "✅ PyInstaller installation completed!"
else
    echo "✅ PyInstaller is already installed."
fi

# Clean previous build
echo "🧹 Cleaning previous build files..."
rm -rf build dist __pycache__

# Build with PyInstaller
echo "🔨 Building with PyInstaller..."
pyinstaller CMYK_Analyzer.spec

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Build completed successfully!"
    echo ""
    echo "📁 Generated files:"
    echo "   - Single executable: dist/CMYK_Analyzer"
    echo "   - App bundle: dist/CMYK_Analyzer.app"
    echo ""
    echo "💡 You can drag the app bundle to Applications folder to install."
    echo "💡 Or double-click in Finder to run."
else
    echo "❌ Build failed."
    exit 1
fi
