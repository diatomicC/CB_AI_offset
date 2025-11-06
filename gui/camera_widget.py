#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera capture widget for capturing images from camera
"""

import cv2
from datetime import datetime
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage


class CameraCaptureWidget(QWidget):
    """Camera capture widget"""
    image_captured = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Camera preview label
        self.preview_label = QLabel("Camera not connected")
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f0f0f0;")
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_camera_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 10px; }")
        
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 10px; }")
        
        button_layout.addWidget(self.start_camera_btn)
        button_layout.addWidget(self.capture_btn)
        
        layout.addWidget(self.preview_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                QMessageBox.warning(self, "Error", "Cannot open camera.")
                return
            self.timer.start(30)  # Update frame every 30ms
            self.start_camera_btn.setText("Stop Camera")
            self.capture_btn.setEnabled(True)
        else:
            self.stop_camera()
            
    def stop_camera(self):
        if self.camera:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.preview_label.setText("Camera not connected")
            self.start_camera_btn.setText("Start Camera")
            self.capture_btn.setEnabled(False)
            
    def update_frame(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # Convert OpenCV BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio)
                self.preview_label.setPixmap(scaled_pixmap)
                
    def capture_image(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # Save to temporary file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_path = f"captured_{timestamp}.png"
                cv2.imwrite(temp_path, frame)
                self.image_captured.emit(temp_path)

