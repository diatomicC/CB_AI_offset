#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMYK Registration & Tilt Analyzer GUI
Main entry point for the GUI application
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from gui.main_window import CMYKAnalyzerGUI


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set font
    font = QFont("Arial", 11)
    app.setFont(font)
    
    # Set global button style for consistency
    app.setStyleSheet("""
        QPushButton {
            font-size: 14px;
            font-weight: bold;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f0f0f0;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
        QPushButton:disabled {
            color: #999;
            background-color: #f5f5f5;
        }
    """)
    
    # Create and show main window
    window = CMYKAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 
