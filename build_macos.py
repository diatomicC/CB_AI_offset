#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS Executable Build Script
Creates macOS apps with icons using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print("✅ PyInstaller is already installed.")
        return True
    except ImportError:
        print("❌ PyInstaller is not installed.")
        return False

def install_pyinstaller():
    """Install PyInstaller."""
    print("📦 Installing PyInstaller...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("✅ PyInstaller installation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyInstaller installation failed: {e}")
        return False

def build_macos_app():
    """Build macOS app."""
    print("🔨 Starting macOS app build...")
    
    # Current directory
    current_dir = Path(__file__).parent
    icon_path = current_dir / "MyIcon.icns"
    main_script = current_dir / "run_gui.py"
    
    # Check icon file
    if not icon_path.exists():
        print(f"⚠️ Icon file not found: {icon_path}")
        print("   Building with default icon.")
        icon_option = ""
    else:
        print(f"✅ Icon file found: {icon_path}")
        icon_option = f"--icon={icon_path}"
    
    # PyInstaller command configuration
    cmd = [
        "pyinstaller",
        "--onefile",                    # Create single executable
        "--windowed",                   # Create GUI app (hide console)
        "--name=CMYK_Analyzer",         # App name
        "--distpath=dist",              # Output directory
        "--workpath=build",             # Work directory
        "--specpath=build",             # Spec file location
        "--clean",                      # Clean previous build
        "--noconfirm",                  # Overwrite existing files without confirmation
    ]
    
    # Add icon if available
    if icon_option:
        cmd.append(icon_option)
    
    # Add main script
    cmd.append(str(main_script))
    
    print("🚀 Executing PyInstaller command...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        # Execute PyInstaller
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build completed successfully!")
        
        # Check output
        if result.stdout:
            print("📋 Build log:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        if e.stderr:
            print("📋 Error log:")
            print(e.stderr)
        return False

def create_app_bundle():
    """Create macOS app bundle."""
    print("📱 Creating macOS app bundle...")
    
    dist_dir = Path("dist")
    app_name = "CMYK_Analyzer"
    
    if not dist_dir.exists():
        print("❌ dist directory not found.")
        return False
    
    # Check executable
    executable = dist_dir / app_name
    if not executable.exists():
        print(f"❌ Executable not found: {executable}")
        return False
    
    # Create app bundle directory
    app_bundle = dist_dir / f"{app_name}.app"
    contents_dir = app_bundle / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    # Create directories
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy executable
    shutil.copy2(executable, macos_dir / app_name)
    
    # Create Info.plist
    info_plist = contents_dir / "Info.plist"
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>{app_name}</string>
    <key>CFBundleIdentifier</key>
    <string>com.cmyk.analyzer</string>
    <key>CFBundleName</key>
    <string>CMYK Analyzer</string>
    <key>CFBundleDisplayName</key>
    <string>CMYK Registration & Tilt Analyzer</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
</dict>
</plist>"""
    
    with open(info_plist, 'w', encoding='utf-8') as f:
        f.write(plist_content)
    
    # Copy icon (if available)
    icon_path = Path("MyIcon.icns")
    if icon_path.exists():
        shutil.copy2(icon_path, resources_dir / "AppIcon.icns")
        print("✅ Icon included in app bundle.")
    
    print(f"✅ App bundle created: {app_bundle}")
    return True

def main():
    """Main function"""
    print("🎯 CMYK Analyzer macOS Executable Builder")
    print("=" * 50)
    
    # Check and install PyInstaller
    if not check_pyinstaller():
        if not install_pyinstaller():
            print("❌ Failed to install PyInstaller.")
            return
    
    # Execute build
    if build_macos_app():
        # Create app bundle
        create_app_bundle()
        
        print("\n🎉 Build completed!")
        print("📁 Output files:")
        print("   - Single executable: dist/CMYK_Analyzer")
        print("   - App bundle: dist/CMYK_Analyzer.app")
        print("\n💡 You can drag the app bundle to Applications folder to install.")
    else:
        print("❌ Build failed.")

if __name__ == "__main__":
    main()
