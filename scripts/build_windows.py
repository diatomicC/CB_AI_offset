#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows Executable Build Script
Creates Windows .exe files using PyInstaller
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

def check_icon():
    """Check if icon file exists."""
    icon_path = Path("../assets/icons/app_icon.ico")
    if icon_path.exists():
        print(f"✅ Icon file found: {icon_path}")
        return True
    else:
        print(f"⚠️ Icon file not found: {icon_path}")
        print("   Building with default icon.")
        return False

def build_windows_exe():
    """Build Windows .exe file."""
    print("🔨 Starting Windows executable build...")
    
    # Current directory
    current_dir = Path(__file__).parent
    spec_file = current_dir / "CMYK_Analyzer_Windows.spec"
    
    if not spec_file.exists():
        print(f"❌ Windows spec file not found: {spec_file}")
        return False
    
    # PyInstaller command configuration
    cmd = [
        "pyinstaller",
        "--clean",                      # Clean previous build
        "--noconfirm",                  # Overwrite existing files without confirmation
        str(spec_file)
    ]
    
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

def create_installer():
    """Create NSIS installer (if NSIS is installed)."""
    print("📦 Attempting to create installer...")
    
    # Check for NSIS
    nsis_path = None
    possible_paths = [
        r"C:\Program Files\NSIS\makensis.exe",
        r"C:\Program Files (x86)\NSIS\makensis.exe",
        "makensis"  # If in PATH
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, "/VERSION"], capture_output=True, text=True)
            if result.returncode == 0:
                nsis_path = path
                print(f"✅ Found NSIS: {path}")
                break
        except FileNotFoundError:
            continue
    
    if not nsis_path:
        print("⚠️ NSIS is not installed. Skipping installer creation.")
        print("   NSIS download: https://nsis.sourceforge.io/Download")
        return False
    
    # Create NSIS script
    nsis_script = """!include "MUI2.nsh"

; Basic settings
Name "CMYK Analyzer"
OutFile "CMYK_Analyzer_Setup.exe"
InstallDir "$PROGRAMFILES\\CMYK Analyzer"
InstallDirRegKey HKCU "Software\\CMYK Analyzer" ""

; Request permissions
RequestExecutionLevel admin

; Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "app_icon.ico"
!define MUI_UNICON "app_icon.ico"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstall pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Languages
!insertmacro MUI_LANGUAGE "Korean"
!insertmacro MUI_LANGUAGE "English"

; Section
Section "CMYK Analyzer" SecMain
    SetOutPath "$INSTDIR"
    
    ; Copy files
    File "dist\\CMYK_Analyzer.exe"
    File "..\\assets\\icons\\app_icon.ico"
    
    ; Create start menu
    CreateDirectory "$SMPROGRAMS\\CMYK Analyzer"
    CreateShortCut "$SMPROGRAMS\\CMYK Analyzer\\CMYK Analyzer.lnk" "$INSTDIR\\CMYK_Analyzer.exe"
    CreateShortCut "$SMPROGRAMS\\CMYK Analyzer\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"
    
    ; Create desktop shortcut
    CreateShortCut "$DESKTOP\\CMYK Analyzer.lnk" "$INSTDIR\\CMYK_Analyzer.exe"
    
    ; Register in registry
    WriteRegStr HKCU "Software\\CMYK Analyzer" "" $INSTDIR
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CMYK Analyzer" "DisplayName" "CMYK Analyzer"
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CMYK Analyzer" "UninstallString" "$INSTDIR\\Uninstall.exe"
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CMYK Analyzer" "DisplayIcon" "$INSTDIR\\app_icon.ico"
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CMYK Analyzer" "Publisher" "CMYK Analyzer Team"
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CMYK Analyzer" "DisplayVersion" "1.0.0"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
SectionEnd

Section "Uninstall"
    ; Delete files
    Delete "$INSTDIR\\CMYK_Analyzer.exe"
    Delete "$INSTDIR\\app_icon.ico"
    Delete "$INSTDIR\\Uninstall.exe"
    
    ; Delete directories
    RMDir "$INSTDIR"
    
    ; Delete start menu
    Delete "$SMPROGRAMS\\CMYK Analyzer\\CMYK Analyzer.lnk"
    Delete "$SMPROGRAMS\\CMYK Analyzer\\Uninstall.lnk"
    RMDir "$SMPROGRAMS\\CMYK Analyzer"
    
    ; Delete desktop shortcut
    Delete "$DESKTOP\\CMYK Analyzer.lnk"
    
    ; Delete registry keys
    DeleteRegKey HKCU "Software\\CMYK Analyzer"
    DeleteRegKey HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CMYK Analyzer"
SectionEnd
"""
    
    # Save NSIS script
    with open("installer.nsi", "w", encoding="utf-8") as f:
        f.write(nsis_script)
    
    # Create LICENSE.txt file (if not exists)
    if not Path("LICENSE.txt").exists():
        with open("LICENSE.txt", "w", encoding="utf-8") as f:
            f.write("CMYK Analyzer License\n====================\n\nThis software is provided as-is for educational and research purposes.\n")
    
    # Execute NSIS
    try:
        print("🔨 Creating NSIS installer...")
        subprocess.run([nsis_path, "installer.nsi"], check=True)
        print("✅ Installer created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installer creation failed: {e}")
        return False

def main():
    """Main function"""
    print("🎯 CMYK Analyzer Windows Executable Builder")
    print("=" * 50)
    
    # Check and install PyInstaller
    if not check_pyinstaller():
        if not install_pyinstaller():
            print("❌ Failed to install PyInstaller.")
            return
    
    # Check icon
    check_icon()
    
    # Execute build
    if build_windows_exe():
        print("\n🎉 Windows build completed!")
        print("📁 Output files:")
        print("   - Executable: dist/CMYK_Analyzer.exe")
        
        # Try to create installer
        if create_installer():
            print("   - Installer: CMYK_Analyzer_Setup.exe")
        
        print("\n💡 Deployment methods:")
        print("   1. Share dist/CMYK_Analyzer.exe directly")
        print("   2. Use installer")
        print("   3. Compress to ZIP file for distribution")
    else:
        print("❌ Build failed.")

if __name__ == "__main__":
    main()
