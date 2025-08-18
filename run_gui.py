#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMYS Registration & Tilt Analyzer GUI Launcher Script

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
    # List of required packages for the application to function properly
    required_packages = [
        'PySide6',        # Modern Qt-based GUI framework
        'opencv-python',  # Computer vision library
        'numpy',          # Numerical computing library
        'Pillow'          # Python Imaging Library
    ]
    
    missing_packages = []
    
    # Iterate through each required package and attempt to import it
    for package in required_packages:
        try:
            # Handle special cases for packages with different import names
            if package == 'opencv-python':
                import cv2
            elif package == 'Pillow':
                import PIL
            else:
                # Standard import for other packages
                __import__(package)
                
        except ImportError:
            # Package is not available, add to missing list
            missing_packages.append(package)
    
    # If any packages are missing, provide user feedback
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
    
    This function performs the following steps:
    1. Displays application header and information
    2. Validates the current working directory
    3. Checks for the existence of the main GUI file
    4. Verifies all dependencies are installed
    5. Launches the GUI application with proper error handling
    """
    # Display application header and welcome message
    print("🎯 CMYS Registration & Tilt Analyzer GUI")
    print("=" * 50)
    
    # Get the absolute path of the current script's directory
    # This ensures we can find the GUI file regardless of where the script is called from
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gui_file = os.path.join(current_dir, "cmyk_analyzer_gui.py")
    
    # Verify that the main GUI file exists in the expected location
    if not os.path.exists(gui_file):
        print(f"❌ GUI file not found at expected location: {gui_file}")
        print("   Please ensure the script is run from the correct project directory.")
        return
    
    # Perform dependency validation before launching the application
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Application cannot start due to missing dependencies.")
        print("   Please install the required packages and try again.")
        return
    
    # All dependencies are available, proceed with launch
    print("✅ All dependencies are properly installed.")
    print("🚀 Launching GUI application...")
    print("-" * 50)
    
    try:
        # Launch the main GUI application using the current Python interpreter
        # This ensures compatibility with the user's Python environment
        subprocess.run([sys.executable, gui_file], check=True)
        
    except subprocess.CalledProcessError as e:
        # Handle errors that occur during the subprocess execution
        print(f"❌ Error occurred while launching GUI: {e}")
        print("   This may indicate an issue with the GUI application itself.")
        
    except KeyboardInterrupt:
        # Handle graceful shutdown when user presses Ctrl+C
        print("\n👋 GUI launch was interrupted by user.")
        print("   Application startup cancelled.")
        
    except Exception as e:
        # Catch any other unexpected errors during the launch process
        print(f"❌ Unexpected error occurred during launch: {e}")
        print("   Please check the error details and try again.")

# Standard Python idiom for running the script directly
if __name__ == "__main__":
    # Execute the main function when the script is run directly
    # This prevents the main function from running if the script is imported as a module
    main() 