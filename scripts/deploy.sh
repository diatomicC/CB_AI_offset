#!/bin/bash

# CMYK Analyzer macOS Deployment Script

echo "🚀 CMYK Analyzer macOS Deployment Started"
echo "=================================================="

# Check build
if [ ! -d "dist" ]; then
    echo "❌ dist directory not found."
    echo "   Please run build first: ./build.sh"
    exit 1
fi

# Get version information
read -p "📝 Enter deployment version (e.g., 1.0.0): " VERSION
read -p "📝 Enter release notes: " RELEASE_NOTES

# Create deployment directory
DEPLOY_DIR="deploy_${VERSION}"
mkdir -p "$DEPLOY_DIR"

echo "📁 Deployment directory created: $DEPLOY_DIR"

# 1. Copy single executable
echo "📦 Copying single executable..."
cp "dist/CMYK_Analyzer" "$DEPLOY_DIR/CMYK_Analyzer_${VERSION}"

# 2. Copy app bundle
echo "📱 Copying app bundle..."
cp -r "dist/CMYK_Analyzer.app" "$DEPLOY_DIR/"

# 3. Create ZIP file
echo "🗜️ Creating ZIP file..."
cd "$DEPLOY_DIR"
zip -r "CMYK_Analyzer_macOS_${VERSION}.zip" "CMYK_Analyzer.app"
cd ..

# 4. Create DMG file (if create-dmg is installed)
if command -v create-dmg &> /dev/null; then
    echo "💿 Creating DMG file..."
    create-dmg \
        --volname "CMYK Analyzer ${VERSION}" \
        --window-pos 200 120 \
        --window-size 800 400 \
        --icon-size 100 \
        --icon "CMYK_Analyzer.app" 200 190 \
        --hide-extension "CMYK_Analyzer.app" \
        --app-drop-link 600 185 \
        "${DEPLOY_DIR}/CMYK_Analyzer_macOS_${VERSION}.dmg" \
        "${DEPLOY_DIR}/"
else
    echo "⚠️ create-dmg is not installed. Skipping DMG creation."
    echo "   Install: brew install create-dmg"
fi

# 5. Create README file
echo "📖 Creating README file..."
cat > "$DEPLOY_DIR/README.txt" << EOF
CMYK Registration & Tilt Analyzer ${VERSION}
==================================================

📱 macOS Executable Distribution

📦 Included files:
- CMYK_Analyzer_${VERSION}: Single executable
- CMYK_Analyzer.app: macOS app bundle
- CMYK_Analyzer_macOS_${VERSION}.zip: App bundle archive
${if command -v create-dmg &> /dev/null; then echo "- CMYK_Analyzer_macOS_${VERSION}.dmg: DMG installer"; fi}

🚀 Installation:

Method 1: Direct app bundle execution
- Double-click CMYK_Analyzer.app to run

Method 2: Install to Applications folder
- Drag CMYK_Analyzer.app to Applications folder

Method 3: Run from terminal
- ./CMYK_Analyzer_${VERSION}

📋 System Requirements:
- macOS 10.15 (Catalina) or later
- Intel or Apple Silicon (M1/M2) support

🔧 Troubleshooting:
- If app doesn't run: Click "Open Anyway" in System Preferences > Security & Privacy
- Permission issues: Use chmod +x command in terminal

📝 Release Notes:
${RELEASE_NOTES}

📞 Support:
- GitHub Issues: [Repository URL]
- Email: [Email Address]

© 2025 CMYK Analyzer Team
EOF

# 6. Generate checksums
echo "🔒 Generating checksums..."
cd "$DEPLOY_DIR"
shasum -a 256 * > "checksums.txt"
cd ..

echo ""
echo "🎉 Deployment completed!"
echo "=================================================="
echo "📁 Deployment directory: $DEPLOY_DIR"
echo "📦 Generated files:"
ls -la "$DEPLOY_DIR"
echo ""
echo "💡 Distribution methods:"
echo "   1. Upload to GitHub Releases"
echo "   2. Share files directly"
echo "   3. Provide download links on website"
echo ""
echo "🚀 Good luck!"
