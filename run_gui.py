#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMYK Registration & Tilt Analyzer GUI Launcher Script

This script serves as the main entry point for launching the CMYK Registration & Tilt Analyzer GUI application.
It performs dependency checking, environment validation, and error handling before launching the main GUI.
"""

import sys
import os
import subprocess

def check_dependencies():
    """
    Check if all required packages are installed and accessible.
    
    This function verifies that the following essential packages are available:
    - PySide6: Qt-based GUI framework
    - opencv-python: Computer vision and image processing
    - numpy: Numerical computing
    - Pillow: Image processing and manipulation
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    required_packages = [
        'PySide6',        # Modern Qt-based GUI framework
        'opencv-python',  # Computer vision library
        'numpy',          # Numerical computing library
        'Pillow'          # Python Imaging Library
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'Pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ The following required packages are not installed:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 To install missing packages, run:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """
    Main function that orchestrates the application launch process.
    
    Steps:
    1. Display application header
    2. Validate current working directory
    3. Check for the existence of the main GUI file
    4. Verify dependencies
    5. Launch GUI application
    """
    print("🎯 CMYK Registration & Tilt Analyzer GUI")
    print("=" * 50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
        gui_file = os.path.join(bundle_dir, "cmyk_analyzer_gui.py")
        print(f"📦 Running inside PyInstaller bundle: {bundle_dir}")
    else:
        gui_file = os.path.join(current_dir, "cmyk_analyzer_gui.py")
    
    if not os.path.exists(gui_file):
        print(f"❌ GUI file not found at expected location: {gui_file}")
        print("   Please ensure the script is run from the correct project directory.")
        print(f"   Current directory: {current_dir}")
        if getattr(sys, 'frozen', False):
            print(f"   Bundle directory: {bundle_dir}")
        return
    
    if getattr(sys, 'frozen', False):
        print("✅ Running inside PyInstaller bundle - skipping dependency check")
        print("🚀 Launching GUI application...")
        print("-" * 50)
    else:
        print("🔍 Checking dependencies...")
        if not check_dependencies():
            print("\n❌ Application cannot start due to missing dependencies.")
            print("   Please install the required packages and try again.")
            return
        
        print("✅ All dependencies are properly installed.")
        print("🚀 Launching GUI application...")
        print("-" * 50)
    
    try:
        if getattr(sys, 'frozen', False):
            print("📱 Executing GUI module directly...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("cmyk_analyzer_gui", gui_file)
            gui_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gui_module)
            
            if hasattr(gui_module, 'main'):
                print("🎯 Executing GUI main function...")
                gui_module.main()
            else:
                print("❌ Could not find 'main' function in GUI module.")
                print("   Available functions:", [attr for attr in dir(gui_module) if not attr.startswith('_')])
        else:
            print("🐍 Running GUI in Python environment...")
            subprocess.run([sys.executable, gui_file], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while launching GUI: {e}")
        print("   This may indicate an issue with the GUI application itself.")
        
    except KeyboardInterrupt:
        print("\n👋 GUI launch was interrupted by user.")
        print("   Application startup cancelled.")
        
    except Exception as e:
        print(f"❌ Unexpected error occurred during launch: {e}")
        print("   Please check the error details and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
