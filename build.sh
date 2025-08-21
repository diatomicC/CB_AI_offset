#!/bin/bash

# CMYK Analyzer Build Script - Main Entry Point

echo "🎯 CMYK Analyzer Build System"
echo "=================================================="

# Check current platform
PLATFORM=$(uname -s)
case $PLATFORM in
    "Darwin")
        echo "🍎 macOS detected - Running macOS build..."
        # PyInstaller로 직접 빌드 (spec 파일 사용)
        echo "🔨 Building with PyInstaller..."
        pyinstaller scripts/CMYK_Analyzer.spec
        if [ $? -eq 0 ]; then
            echo "✅ macOS build completed!"
            echo "📁 Generated files:"
            echo "   - Single executable: dist/CMYK_Analyzer"  
            echo "   - App bundle: dist/CMYK_Analyzer.app"
        else
            echo "❌ macOS build failed."
        fi
        ;;
    "Linux")
        echo "🐧 Linux detected - Running Docker build for Windows..."
        ./scripts/build_windows_docker.sh
        ;;
    *)
        echo "❓ Unknown platform: $PLATFORM"
        echo "   Available build options:"
        echo "   - python scripts/build_macos.py (macOS)"
        echo "   - ./scripts/build_windows_docker.sh (Windows via Docker)"
        echo "   - ./scripts/build_windows_simple.sh (Simple Windows build)"
        ;;
esac
