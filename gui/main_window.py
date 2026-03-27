#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main window GUI for CMYK Registration Analyzer
Contains the main application window and UI components
"""

import os
import json
import cv2
from datetime import datetime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox, 
    QGridLayout, QMessageBox, QProgressBar, QSplitter, 
    QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont, QColor

from gui.analysis_worker import AnalysisWorker
from gui.camera_widget import CameraCaptureWidget


class CMYKAnalyzerGUI(QMainWindow):
    """Main GUI application"""
    
    def __init__(self):
        super().__init__()
        self.left_image_path = None
        self.right_image_path = None
        self.analysis_results = None
        self.worker = None
        self.camera_capture_count = 0
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Registration Analyzer (K-frame, position-based)")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 900])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        self.statusBar().showMessage("Ready")
    
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Image input
        input_group = QGroupBox("Image Input (Left & Right)")
        input_layout = QVBoxLayout()
        
        upload_buttons_layout = QHBoxLayout()
        self.upload_left_btn = QPushButton("Upload Left Image")
        self.upload_left_btn.clicked.connect(self.upload_left_image)
        upload_buttons_layout.addWidget(self.upload_left_btn)
        
        self.upload_right_btn = QPushButton("Upload Right Image")
        self.upload_right_btn.clicked.connect(self.upload_right_image)
        upload_buttons_layout.addWidget(self.upload_right_btn)
        input_layout.addLayout(upload_buttons_layout)
        
        status_layout = QHBoxLayout()
        self.left_image_status = QLabel("No Left Image")
        self.right_image_status = QLabel("No Right Image")
        status_layout.addWidget(self.left_image_status)
        status_layout.addWidget(self.right_image_status)
        input_layout.addLayout(status_layout)
        
        # Camera
        self.camera_widget = CameraCaptureWidget()
        self.camera_widget.image_captured.connect(self.handle_camera_capture)
        input_layout.addWidget(self.camera_widget)
        
        # Preview
        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout()
        
        left_preview_layout = QHBoxLayout()
        left_preview_layout.addWidget(QLabel("Left:"))
        self.left_preview_label = QLabel("No image")
        self.left_preview_label.setMinimumSize(150, 100)
        self.left_preview_label.setMaximumSize(200, 150)
        self.left_preview_label.setAlignment(Qt.AlignCenter)
        self.left_preview_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f0f0f0; color: #666; font-size: 10px;")
        left_preview_layout.addWidget(self.left_preview_label)
        preview_layout.addLayout(left_preview_layout)
        
        right_preview_layout = QHBoxLayout()
        right_preview_layout.addWidget(QLabel("Right:"))
        self.right_preview_label = QLabel("No image")
        self.right_preview_label.setMinimumSize(150, 100)
        self.right_preview_label.setMaximumSize(200, 150)
        self.right_preview_label.setAlignment(Qt.AlignCenter)
        self.right_preview_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f0f0f0; color: #666; font-size: 10px;")
        right_preview_layout.addWidget(self.right_preview_label)
        preview_layout.addLayout(right_preview_layout)
        
        preview_group.setLayout(preview_layout)
        input_layout.addWidget(preview_group)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Analyze
        self.analyze_btn = QPushButton("Analyze Left Image")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)
        
        # Progress + status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Upload left image to enable analysis. Right image is optional.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        self.result_tabs = QTabWidget()
        
        # Adjustment values tab (Main tab)
        self.adjustment_tab = QWidget()
        self.adjustment_layout = QVBoxLayout()
        
        title_label = QLabel("PRINTING MACHINE ADJUSTMENT VALUES")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        self.adjustment_layout.addWidget(title_label)
        
        instructions_label = QLabel("Use these values to adjust your printing machine:")
        instructions_label.setStyleSheet("font-size: 14px; color: #34495e; margin: 5px;")
        instructions_label.setAlignment(Qt.AlignCenter)
        self.adjustment_layout.addWidget(instructions_label)
        
        # Strategy info label
        self.strategy_label = QLabel("Strategy: Analyzing...")
        self.strategy_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin: 5px; font-style: italic;")
        self.strategy_label.setAlignment(Qt.AlignCenter)
        self.strategy_label.setWordWrap(True)
        self.adjustment_layout.addWidget(self.strategy_label)
        
        # Adjustment table
        self.adjustment_table = QTableWidget()
        self.adjustment_table.setRowCount(5)  # C, M, Y, K, S
        self.adjustment_table.setColumnCount(4)  # LATERAL, CIRCUM, COCKING, Status
        self.adjustment_table.setHorizontalHeaderLabels(['LATERAL (mm)', 'CIRCUM (mm)', 'COCKING (mm)', 'Status'])
        
        row_labels = [
            'Upper Left',
            'Upper Right',
            'Lower Left',
            'Lower Right',
            'K (Frame)',
        ]
        self.adjustment_table.setVerticalHeaderLabels(row_labels)
        
        self.adjustment_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #bdc3c7;
                font-size: 14px;
                background-color: white;
                alternate-background-color: #ecf0f1;
                selection-background-color: #3498db;
            }
            QTableWidget::item {
                padding: 8px;
                text-align: center;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border: none;
            }
        """)
        
        self.adjustment_table.setAlternatingRowColors(True)
        self.adjustment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.adjustment_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.adjustment_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.adjustment_layout.addWidget(self.adjustment_table)
        
        # Instructions group
        instructions_group = QGroupBox("Adjustment Instructions")
        instructions_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        instructions_layout = QVBoxLayout()
        
        instructions_text = """
• LATERAL: Adjust left-right position
  - Positive value: Move RIGHT
  - Negative value: Move LEFT

• CIRCUM: Adjust up-down position
  - Positive value: Move UP
  - Negative value: Move DOWN

• COCKING: Y coordinate difference (Right - Left)
  - Positive value: Right image is LOWER than Left
  - Negative value: Right image is HIGHER than Left

TIP: Make small adjustments and re-analyze for best results.
        """
        
        instructions_detail = QLabel(instructions_text.strip())
        instructions_detail.setStyleSheet("font-size: 12px; color: #2c3e50; margin: 10px;")
        instructions_detail.setWordWrap(True)
        instructions_layout.addWidget(instructions_detail)
        instructions_group.setLayout(instructions_layout)
        
        self.adjustment_layout.addWidget(instructions_group)
        self.adjustment_tab.setLayout(self.adjustment_layout)
        self.result_tabs.addTab(self.adjustment_tab, "Adjustment Values")
        
        # Visualization tab
        self.visualization_tab = QWidget()
        self.visualization_layout = QVBoxLayout()
        
        image_btn_layout = QGridLayout()
        self.show_preprocessed_btn = QPushButton("1. Preprocessed")
        self.show_preprocessed_btn.clicked.connect(lambda: self.show_debug_image('preprocessed'))
        self.show_cmyk_btn = QPushButton("2. Face detection")
        self.show_cmyk_btn.clicked.connect(lambda: self.show_debug_image('cmyk_detection'))
        self.show_pivot_btn = QPushButton("3. Pivot")
        self.show_pivot_btn.clicked.connect(lambda: self.show_debug_image('pivot'))
        self.show_t_points_btn = QPushButton("4. T Points")
        self.show_t_points_btn.clicked.connect(lambda: self.show_debug_image('t_points'))
        self.show_cocking_btn = QPushButton("5. Cocking")
        self.show_cocking_btn.clicked.connect(lambda: self.show_debug_image('cocking'))
        self.show_adjusted_btn = QPushButton("6. After Adjustment")
        self.show_adjusted_btn.clicked.connect(lambda: self.show_debug_image('after_adjustment'))
        image_btn_layout.addWidget(self.show_preprocessed_btn, 0, 0)
        image_btn_layout.addWidget(self.show_cmyk_btn, 0, 1)
        image_btn_layout.addWidget(self.show_pivot_btn, 0, 2)
        image_btn_layout.addWidget(self.show_t_points_btn, 1, 0)
        image_btn_layout.addWidget(self.show_cocking_btn, 1, 1)
        image_btn_layout.addWidget(self.show_adjusted_btn, 1, 2)
        self.visualization_layout.addLayout(image_btn_layout)
        
        self.image_scroll = QScrollArea()
        self.image_label = QLabel("Select an image type above to view analysis results")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(True)
        self.visualization_layout.addWidget(self.image_scroll)
        self.visualization_tab.setLayout(self.visualization_layout)
        self.result_tabs.addTab(self.visualization_tab, "Visualization")
        
        layout.addWidget(self.result_tabs)
        
        # Save buttons
        save_layout = QHBoxLayout()
        self.save_json_btn = QPushButton("Save JSON")
        self.save_json_btn.clicked.connect(self.save_json)
        self.save_json_btn.setEnabled(False)
        self.save_image_btn = QPushButton("Save Image")
        self.save_image_btn.clicked.connect(self.save_debug_image)
        self.save_image_btn.setEnabled(False)
        save_layout.addWidget(self.save_json_btn)
        save_layout.addWidget(self.save_image_btn)
        layout.addLayout(save_layout)
        
        panel.setLayout(layout)
        return panel
    
    def upload_left_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Left Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            self.load_left_image(file_path)
    
    def upload_right_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Right Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            self.load_right_image(file_path)
    
    def handle_camera_capture(self, image_path):
        if self.camera_capture_count % 2 == 0:
            self.load_left_image(image_path)
        else:
            self.load_right_image(image_path)
        self.camera_capture_count += 1
    
    def load_left_image(self, image_path):
        self.left_image_path = image_path
        self.left_image_status.setText(f"Left: {os.path.basename(image_path)}")
        self.update_image_preview(image_path, self.left_preview_label, "Left")
        self.update_analysis_button_state()
        self.statusBar().showMessage(f"Left image loaded: {os.path.basename(image_path)}")
    
    def load_right_image(self, image_path):
        self.right_image_path = image_path
        self.right_image_status.setText(f"Right: {os.path.basename(image_path)}")
        self.update_image_preview(image_path, self.right_preview_label, "Right")
        self.statusBar().showMessage(f"Right image loaded: {os.path.basename(image_path)}")
    
    def update_image_preview(self, image_path, preview_label, image_type):
        try:
            img = cv2.imread(image_path)
            if img is None:
                preview_label.setText(f"Failed to load\n{image_type} image")
                return
            h, w = img.shape[:2]
            preview_w, preview_h = 150, 100
            aspect = w / h
            if aspect > preview_w / preview_h:
                new_w = preview_w
                new_h = int(preview_w / aspect)
            else:
                new_h = preview_h
                new_w = int(preview_h * aspect)
            resized = cv2.resize(img, (new_w, new_h))
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            preview_label.setPixmap(pixmap)
            preview_label.setScaledContents(False)
            preview_label.setAlignment(Qt.AlignCenter)
        except Exception as e:
            preview_label.setText(f"Preview error\n{image_type}")
    
    def update_analysis_button_state(self):
        if self.left_image_path:
            self.analyze_btn.setEnabled(True)
            self.status_label.setText("Ready to analyze! Click 'Analyze Left Image'.")
        else:
            self.analyze_btn.setEnabled(False)
            self.status_label.setText("Upload left image to enable analysis. Right image is optional.")
    
    def start_analysis(self):
        if not self.left_image_path:
            QMessageBox.warning(self, "Error", "Please load left image first.")
            return
        
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Analyzing left image...")
        self.analyze_btn.setEnabled(False)
        
        self.worker = AnalysisWorker(self.left_image_path, self.right_image_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.start()
    
    def update_progress(self, message):
        self.status_label.setText(message)
        self.statusBar().showMessage(message)
    
    def analysis_finished(self, results):
        self.analysis_results = results
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Analysis completed.")
        self.display_results(results)
        self.save_json_btn.setEnabled(True)
        self.save_image_btn.setEnabled(True)
        self.statusBar().showMessage("Analysis completed")
    
    def analysis_error(self, error_message):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.statusBar().showMessage("Analysis failed")
    
    def display_results(self, results):
        # Populate adjustment table
        adjustments = results.get('adjustments', {})
        self.populate_adjustment_table(adjustments)
        
        # Switch to Adjustment Values tab (main tab)
        self.result_tabs.setCurrentIndex(0)
        
        # Show default visualization
        self.show_debug_image('preprocessed')
    
    def populate_adjustment_table(self, adjustments):
        """Populate the adjustment table with results"""
        # Extract strategy info if available
        strategy_info = adjustments.pop('_strategy_info', None)
        if strategy_info:
            strategy = strategy_info.get('strategy', 'individual')
            shift_analysis = strategy_info.get('shift_analysis', {})
            
            if strategy == 'global':
                common_shift = shift_analysis.get('common_shift', (0, 0))
                consistency = shift_analysis.get('shift_consistency', 0)
                optimal_shift = strategy_info.get('optimal_common_shift', common_shift)
                global_cost = strategy_info.get('global_cost', 0)
                individual_cost = strategy_info.get('individual_cost', 0)
                frame_shift = shift_analysis.get('frame_center_shift', optimal_shift)
                strategy_text = f"Strategy: GLOBAL SHIFT (K frame moves, then individual fine-tuning)\nK frame shift: LATERAL={optimal_shift[0]:.3f}mm, CIRCUM={optimal_shift[1]:.3f}mm | Total cost: {global_cost:.3f}mm (vs Individual: {individual_cost:.3f}mm)"
            else:
                individual_cost = strategy_info.get('individual_cost', 0)
                strategy_text = f"Strategy: INDIVIDUAL CORRECTION (Each color adjusted separately)\nTotal cost: {individual_cost:.3f}mm"
            
            self.strategy_label.setText(strategy_text)
            self.strategy_label.setStyleSheet("font-size: 12px; color: #2980b9; margin: 5px; font-weight: bold;")
        else:
            self.strategy_label.setText("Strategy: INDIVIDUAL CORRECTION")
            self.strategy_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin: 5px;")
        
        # Clear existing items first
        self.adjustment_table.clearContents()
        
        row_order = ['upper_left', 'upper_right', 'lower_left', 'lower_right', 'K']
        for row, key in enumerate(row_order):
            adj = adjustments.get(key, {})
            
            # LATERAL
            lateral_val = adj.get('LATERAL')
            if lateral_val is not None:
                lateral_item = QTableWidgetItem(f"{lateral_val:+.3f}")
                lateral_item.setTextAlignment(Qt.AlignCenter)
                if lateral_val > 0:
                    lateral_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                elif lateral_val < 0:
                    lateral_item.setBackground(QColor(235, 255, 235))  # Light green for negative
                else:
                    lateral_item.setBackground(QColor(255, 255, 255))  # White for zero
            else:
                lateral_item = QTableWidgetItem("N/A")
                lateral_item.setTextAlignment(Qt.AlignCenter)
                lateral_item.setBackground(QColor(240, 240, 240))  # Gray for N/A
            
            # CIRCUM
            circum_val = adj.get('CIRCUM')
            if circum_val is not None:
                circum_item = QTableWidgetItem(f"{circum_val:+.3f}")
                circum_item.setTextAlignment(Qt.AlignCenter)
                if circum_val > 0:
                    circum_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                elif circum_val < 0:
                    circum_item.setBackground(QColor(235, 255, 235))  # Light green for negative
                else:
                    circum_item.setBackground(QColor(255, 255, 255))  # White for zero
            else:
                circum_item = QTableWidgetItem("N/A")
                circum_item.setTextAlignment(Qt.AlignCenter)
                circum_item.setBackground(QColor(240, 240, 240))  # Gray for N/A
            
            # COCKING
            cocking_val = adj.get('COCKING')
            if cocking_val is not None:
                cocking_item = QTableWidgetItem(f"{cocking_val:+.3f}")
                cocking_item.setTextAlignment(Qt.AlignCenter)
                if cocking_val > 0:
                    cocking_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                elif cocking_val < 0:
                    cocking_item.setBackground(QColor(235, 255, 235))  # Light green for negative
                else:
                    cocking_item.setBackground(QColor(255, 255, 255))  # White for zero
            else:
                cocking_item = QTableWidgetItem("N/A")
                cocking_item.setTextAlignment(Qt.AlignCenter)
                cocking_item.setBackground(QColor(240, 240, 240))  # Gray for N/A
            
            # Status
            status_text = adj.get('status', 'Unknown')
            if status_text == 'OK':
                status_item = QTableWidgetItem("OK")
                status_item.setBackground(QColor(235, 255, 235))  # Light green
            elif status_text == 'Cocking data missing':
                status_item = QTableWidgetItem("Cocking Missing")
                status_item.setBackground(QColor(255, 255, 200))  # Light yellow
            elif status_text == 'Warning':
                status_item = QTableWidgetItem("Warning")
                status_item.setBackground(QColor(255, 235, 200))  # Light orange
            elif status_text == 'N/A (Individual mode)':
                status_item = QTableWidgetItem("N/A")
                status_item.setBackground(QColor(245, 245, 245))  # Light gray
            else:
                status_item = QTableWidgetItem(status_text)
                status_item.setBackground(QColor(255, 220, 220))  # Light red
            
            status_item.setTextAlignment(Qt.AlignCenter)
            
            # Set items in table
            self.adjustment_table.setItem(row, 0, lateral_item)
            self.adjustment_table.setItem(row, 1, circum_item)
            self.adjustment_table.setItem(row, 2, cocking_item)
            self.adjustment_table.setItem(row, 3, status_item)
        
        # Resize table to fit contents
        self.adjustment_table.resizeRowsToContents()
    
    def show_debug_image(self, image_type):
        if not self.analysis_results:
            self.image_label.setText("No analysis results available")
            self.image_label.update()
            return
        if image_type == 'preprocessed':
            debug_img = self.analysis_results.get('preprocessed')
        elif image_type == 'cmyk_detection':
            debug_img = self.analysis_results.get('cmyk_detection')
        elif image_type == 'pivot':
            debug_img = self.analysis_results.get('pivot')
        elif image_type == 't_points':
            debug_img = self.analysis_results.get('t_points')
        elif image_type == 'cocking':
            debug_img = self.analysis_results.get('cocking_debug')
            if debug_img is None:
                if self.right_image_path:
                    self.image_label.setText("Cocking analysis failed (Right image processing error)")
                else:
                    self.image_label.setText("Cocking analysis not available (No right image)")
                return
        elif image_type == 'after_adjustment':
            debug_img = self.analysis_results.get('after_adjustment')
            if debug_img is None:
                self.image_label.setText("After adjustment visualization not available")
                return
        else:
            return
        
        if debug_img is None:
            self.image_label.setText("No image for this view")
            return
        rgb_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.update()
    
    def save_json(self):
        if not self.analysis_results:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save JSON", f"cmyk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "JSON Files (*.json)")
        if file_path:
            try:
                json_results = {}
                for key, value in self.analysis_results.items():
                    if key in ['debug_reg', 'cocking_debug', 'preprocessed', 'cmyk_detection', 'pivot', 't_points']:
                        continue
                    json_results[key] = value
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_results, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "Save Complete", f"Results saved:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error occurred during save:\n{str(e)}")
    
    def save_debug_image(self):
        if not self.analysis_results:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"cmyk_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", "PNG Images (*.png)")
        if file_path:
            try:
                current_pixmap = self.image_label.pixmap()
                if current_pixmap:
                    current_pixmap.save(file_path)
                    QMessageBox.information(self, "Save Complete", f"Image saved:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error occurred during save:\n{str(e)}")
    
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        if hasattr(self, 'camera_widget'):
            self.camera_widget.stop_camera()
        event.accept()
