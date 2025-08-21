#!/bin/bash

# Windows Executable Build Script using Docker

echo "🐳 Windows Build using Docker Started"
echo "=================================================="

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "   Install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check Docker running status
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running."
    echo "   Start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running normally."

# Clean previous build
echo "🧹 Cleaning previous build files..."
rm -rf build dist __pycache__

# Build Docker image
echo "🔨 Building Docker image..."
docker build -f Dockerfile.windows -t cmyk-windows-builder .

if [ $? -ne 0 ]; then
    echo "❌ Docker image build failed."
    exit 1
fi

echo "✅ Docker image build completed!"

# Build Windows executable
echo "🚀 Building Windows executable..."
docker run --rm \
    -v "$(pwd)/dist:/workspace/dist" \
    -v "$(pwd)/build:/workspace/build" \
    cmyk-windows-builder

if [ $? -ne 0 ]; then
    echo "❌ Windows build failed."
    exit 1
fi

# Check build results
echo ""
echo "🎉 Windows build completed!"
echo "=================================================="

if [ -f "dist/CMYK_Analyzer.exe" ]; then
    echo "✅ Windows executable created!"
    echo "📁 File location: dist/CMYK_Analyzer.exe"
    echo "📊 File size: $(ls -lh dist/CMYK_Analyzer.exe | awk '{print $5}')"
    
    # Prepare for deployment
    echo ""
    echo "📦 Preparing for deployment..."
    mkdir -p "deploy_windows"
    cp "dist/CMYK_Analyzer.exe" "deploy_windows/"
    
    if [ -f "app_icon.ico" ]; then
        cp "app_icon.ico" "deploy_windows/"
        echo "✅ Icon file copy completed"
    fi
    
    # Create ZIP file
    cd "deploy_windows"
    zip -r "CMYK_Analyzer_Windows.zip" *
    cd ..
    
    echo "✅ Deployment package created!"
    echo "📁 Deployment directory: deploy_windows/"
    echo "📦 Included files:"
    ls -la "deploy_windows/"
    
else
    echo "❌ Windows executable not found."
    echo "📁 dist/ directory contents:"
    ls -la dist/ 2>/dev/null || echo "   dist/ directory does not exist."
fi

echo ""
echo "💡 Next steps:"
echo "   1. Distribute files from deploy_windows/ directory to Windows users"
echo "   2. Test execution in Windows environment"
echo "   3. If there are issues, check Docker logs: docker logs <container_id>"
