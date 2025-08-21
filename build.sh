#!/bin/bash

# CMYK Analyzer Build Script - Main Entry Point

echo "🎯 CMYK Analyzer Build System"
echo "=================================================="

# Check current platform
PLATFORM=$(uname -s)
case $PLATFORM in
    "Darwin")
        echo "🍎 macOS detected - Running macOS build..."
        ./scripts/build.sh
        ;;
    "Linux")
        echo "🐧 Linux detected - Running Docker build for Windows..."
        ./scripts/build_windows_docker.sh
        ;;
    *)
        echo "❓ Unknown platform: $PLATFORM"
        echo "   Available build scripts:"
        echo "   - ./scripts/build.sh (macOS)"
        echo "   - ./scripts/build_windows_docker.sh (Windows via Docker)"
        echo "   - ./scripts/build_windows_simple.sh (Simple Windows build)"
        ;;
esac
