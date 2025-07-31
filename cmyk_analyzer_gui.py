#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMYK Registration & Tilt Analyzer GUI
산업용 인쇄 품질 관리 및 보정을 위한 CMYK 컬러 박스 정렬 및 기울기 분석 GUI 애플리케이션
"""

import sys
import os
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QSpinBox, QDoubleSpinBox,
    QTextEdit, QGroupBox, QGridLayout, QMessageBox, QProgressBar,
    QSplitter, QFrame, QScrollArea, QSizePolicy, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor

# 기존 분석 모듈 import
from color_registration_analysis import (
    extract_marker, detect_bottom_left, detect_square_corners,
    pixel_to_bottom_left_coord, calculate_tilt_angle, 
    calculate_horizontal_correction, order_points
)

def extract_robust_square_marker(img: np.ndarray) -> np.ndarray | None:
    """
    Detects the largest square marker in the image and extracts it with perspective transform.
    Enhanced version with multiple detection methods.
    
    Args:
        img (np.ndarray): Input image (BGR)
        
    Returns:
        np.ndarray | None: Extracted square region. None if not found
    """
    # Try multiple detection methods
    methods = [
        lambda: _detect_square_method1(img),  # Original method
        lambda: _detect_square_method2(img),  # Adaptive threshold
        lambda: _detect_square_method3(img),  # Edge detection
    ]
    
    for method in methods:
        try:
            result = method()
            if result is not None:
                return result
        except:
            continue
    
    return None

def _detect_square_method1(img: np.ndarray) -> np.ndarray | None:
    """Original detection method"""
    # Step 1: Convert to grayscale and apply Otsu thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 2: Find external contours on inverted binary image
    contours, _ = cv2.findContours(255 - bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000:  # Reduced minimum area
            continue
            
        epsilon = 0.05 * cv2.arcLength(cnt, True)  # Increased tolerance
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Check for convex quadrilaterals (approximate square)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = approx.reshape(4, 2)
            width = np.linalg.norm(rect[0] - rect[1])
            height = np.linalg.norm(rect[1] - rect[2])
            ratio = width / height if height != 0 else 0
            
            if abs(1 - ratio) < 0.5:  # Increased aspect ratio tolerance
                candidates.append((area, rect))
    
    if not candidates:
        return None
    
    # Step 3: Select the largest square candidate
    _, best_rect = max(candidates, key=lambda x: x[0])
    
    # Step 4: Order corners (top-left, top-right, bottom-right, bottom-left)
    def order_points_preprocess(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect = np.zeros((4, 2), dtype='float32')
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    rect = order_points_preprocess(best_rect)
    
    # Step 5: Apply perspective transform to get top-down view
    size = int(max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[1] - rect[2]),
        np.linalg.norm(rect[2] - rect[3]),
        np.linalg.norm(rect[3] - rect[0])
    ))
    
    dst_pts = np.array([
        [0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]
    ], dtype='float32')
    
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(img, M, (size, size))
    
    return warped

def _detect_square_method2(img: np.ndarray) -> np.ndarray | None:
    """Adaptive threshold method"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use adaptive threshold instead of Otsu
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(255 - bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:  # Even smaller minimum area
            continue
            
        epsilon = 0.08 * cv2.arcLength(cnt, True)  # More tolerance
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = approx.reshape(4, 2)
            width = np.linalg.norm(rect[0] - rect[1])
            height = np.linalg.norm(rect[1] - rect[2])
            ratio = width / height if height != 0 else 0
            
            if abs(1 - ratio) < 0.6:  # More aspect ratio tolerance
                candidates.append((area, rect))
    
    if not candidates:
        return None
    
    _, best_rect = max(candidates, key=lambda x: x[0])
    
    def order_points_preprocess(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect = np.zeros((4, 2), dtype='float32')
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    rect = order_points_preprocess(best_rect)
    
    size = int(max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[1] - rect[2]),
        np.linalg.norm(rect[2] - rect[3]),
        np.linalg.norm(rect[3] - rect[0])
    ))
    
    dst_pts = np.array([
        [0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]
    ], dtype='float32')
    
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(img, M, (size, size))
    
    return warped

def _detect_square_method3(img: np.ndarray) -> np.ndarray | None:
    """Edge detection method"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:  # Smallest minimum area
            continue
            
        epsilon = 0.1 * cv2.arcLength(cnt, True)  # Most tolerance
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = approx.reshape(4, 2)
            width = np.linalg.norm(rect[0] - rect[1])
            height = np.linalg.norm(rect[1] - rect[2])
            ratio = width / height if height != 0 else 0
            
            if abs(1 - ratio) < 0.7:  # Most aspect ratio tolerance
                candidates.append((area, rect))
    
    if not candidates:
        return None
    
    _, best_rect = max(candidates, key=lambda x: x[0])
    
    def order_points_preprocess(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect = np.zeros((4, 2), dtype='float32')
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    rect = order_points_preprocess(best_rect)
    
    size = int(max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[1] - rect[2]),
        np.linalg.norm(rect[2] - rect[3]),
        np.linalg.norm(rect[3] - rect[0])
    ))
    
    dst_pts = np.array([
        [0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]
    ], dtype='float32')
    
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(img, M, (size, size))
    
    return warped

def calculate_adjustment_values(registration_results, tilt_results):
    """
    Calculate LATERAL, CIRCUM, COCKING adjustment values for each color.
    
    LATERAL: Left-right movement (X-axis)
    CIRCUM: Up-down movement (Y-axis) 
    COCKING: Rotation around a point (tilt correction)
    
    Args:
        registration_results: Color registration analysis results
        tilt_results: Tilt analysis results
        
    Returns:
        dict: Adjustment values for each color
    """
    adjustments = {}
    
    for color in ['C', 'M', 'Y', 'K']:
        # Get registration data
        reg_data = registration_results.get(color)
        tilt_data = tilt_results.get(color)
        
        if reg_data is None or tilt_data is None:
            adjustments[color] = {
                'LATERAL': None,
                'CIRCUM': None, 
                'COCKING': None,
                'status': 'Detection failed'
            }
            continue
            
        # Extract movement values (in mm)
        dx_mm, dy_mm = reg_data['movement_mm']
        
        # Extract tilt correction (in μm)
        cocking_um = tilt_data['correction_actual_um']
        
        # Convert to practical adjustment values
        # LATERAL = X-axis movement (좌우)
        # Right = +, Left = -
        lateral_mm = dx_mm
        
        # CIRCUM = Y-axis movement (상하)
        # Up = +, Down = -
        circum_mm = dy_mm
        
        # COCKING = tilt correction (μm → mm 변환)
        cocking_mm = cocking_um / 1000.0  # Convert μm to mm
        
        adjustments[color] = {
            'LATERAL': round(lateral_mm, 3),
            'CIRCUM': round(circum_mm, 3),
            'COCKING': round(cocking_mm, 3),
            'status': 'OK'
        }
    
    return adjustments

class AnalysisWorker(QThread):
    """Worker thread that executes analysis tasks in the background"""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, image_path, print_width_mm):
        super().__init__()
        self.image_path = image_path
        self.print_width_mm = print_width_mm
        
    def run(self):
        try:
            self.progress.emit("Loading image...")
            
            # Load image
            orig = cv2.imread(self.image_path)
            if orig is None:
                self.error.emit("Cannot read image file.")
                return
                
            self.progress.emit("Preprocessing square marker...")
            
            # Step 1: Square marker preprocessing (preprocess.ipynb logic)
            preprocessed = extract_robust_square_marker(orig)
            if preprocessed is None:
                self.error.emit("Cannot find square marker. Please ensure the image contains a clear square region.")
                return
                
            self.progress.emit("Extracting CMYK marker region...")
            
            # Step 2: CMYK marker region extraction (existing logic)
            cropped = extract_marker(preprocessed)
            if cropped is None:
                self.error.emit("Cannot find CMYK marker.")
                return
                
            h_px, w_px = cropped.shape[:2]
            mm_per_pixel_x = 5.0 / w_px
            mm_per_pixel_y = 5.0 / h_px
            
            # HSV color ranges
            HSV = {
                'C': ((90,80,80),(130,255,255)),   # Cyan
                'M': ((130,50,70),(170,255,255)),  # Magenta 
                'Y': ((15,60,60),(45,255,255)),    # Yellow - expanded range for better detection
                'K': ((0,0,0),(180,255,50))        # blacK
            }
            
            # Target coordinates (bottom-left origin)
            target_coords = {
                'K': (w_px/10, h_px - h_px*6/10),
                'C': (w_px*6/10, h_px - h_px*6/10),
                'M': (w_px/10, h_px - h_px/10),
                'Y': (w_px*6/10, h_px - h_px/10)
            }
            
            self.progress.emit("Analyzing color registration...")
            
            # 1. Color registration analysis
            results_reg = {}
            debug_reg = cropped.copy()
            
            # Mark target points with red X
            for color, (tx_px, ty_px) in target_coords.items():
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                tx_cv = int(tx_bl)
                ty_cv = int(h_px - ty_bl)
                
                cv2.line(debug_reg, (tx_cv-10, ty_cv-10), (tx_cv+10, ty_cv+10), (0,0,255), 2)
                cv2.line(debug_reg, (tx_cv-10, ty_cv+10), (tx_cv+10, ty_cv-10), (0,0,255), 2)
                cv2.putText(debug_reg, f"T{color}", (tx_cv+12, ty_cv), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            for color, hsv_range in HSV.items():
                bl = detect_bottom_left(cropped, hsv_range, min_area_ratio=0.7)  # 더 엄격한 필터링
                if bl is None:
                    results_reg[color] = None
                    continue
                
                px_px, py_px = bl
                px_bl, py_bl = pixel_to_bottom_left_coord(px_px, py_px, h_px)
                
                tx_px, ty_px = target_coords[color]
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                
                dx_px = tx_bl - px_bl
                dy_px = ty_bl - py_bl
                
                px_mm = px_bl * mm_per_pixel_x
                py_mm = py_bl * mm_per_pixel_y
                tx_mm = tx_bl * mm_per_pixel_x
                ty_mm = ty_bl * mm_per_pixel_y
                dx_mm = dx_px * mm_per_pixel_x
                dy_mm = dy_px * mm_per_pixel_y
                
                results_reg[color] = {
                    'P_coord_mm': (round(px_mm, 3), round(py_mm, 3)),
                    'T_coord_mm': (round(tx_mm, 3), round(ty_mm, 3)),
                    'movement_mm': (round(dx_mm, 3), round(dy_mm, 3))
                }
                
                # Debug visualization
                px_int, py_int = int(px_px), int(py_px)
                cv2.circle(debug_reg, (px_int, py_int), 8, (0,255,0), -1)
                cv2.putText(debug_reg, f"P{color}({px_mm:.2f},{py_mm:.2f})", 
                           (px_int+15, py_int-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(debug_reg, f"Move({dx_mm:.2f},{dy_mm:.2f})", 
                           (px_int+15, py_int+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            self.progress.emit("Analyzing tilt...")
            
            # 2. Tilt analysis
            results_tilt = {}
            debug_tilt = cropped.copy()
            
            for color, hsv_range in HSV.items():
                corners = detect_square_corners(cropped, hsv_range, min_area_ratio=0.7)  # 더 엄격한 필터링
                if corners is None:
                    results_tilt[color] = None
                    continue
                
                tl, tr, br, bl = corners
                
                bl_coord = pixel_to_bottom_left_coord(bl[0], bl[1], h_px)
                br_coord = pixel_to_bottom_left_coord(br[0], br[1], h_px)
                
                angle_deg, dx_px, dy_px = calculate_tilt_angle(bl_coord, br_coord)
                square_width_px = np.linalg.norm(br - bl)
                correction_um = calculate_horizontal_correction(bl_coord, br_coord, square_width_px, mm_per_pixel_x)
                final_correction_um = (correction_um * self.print_width_mm) / 5.0
                
                results_tilt[color] = {
                    'bottom_left_px': [float(bl[0]), float(bl[1])],
                    'bottom_right_px': [float(br[0]), float(br[1])],
                    'bottom_left_coord': [float(bl_coord[0]), float(bl_coord[1])],
                    'bottom_right_coord': [float(br_coord[0]), float(br_coord[1])],
                    'tilt_angle_deg': round(float(angle_deg), 6),
                    'dx_px': round(float(dx_px), 3),
                    'dy_px': round(float(dy_px), 3),
                    'correction_5mm_um': round(float(correction_um), 3),
                    'correction_actual_um': round(float(final_correction_um), 3)
                }
                
                # Debug visualization
                cv2.circle(debug_tilt, (int(bl[0]), int(bl[1])), 8, (0,0,255), -1)
                cv2.putText(debug_tilt, f"{color}_BL", (int(bl[0])+10, int(bl[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
                cv2.circle(debug_tilt, (int(br[0]), int(br[1])), 8, (255,0,0), -1)
                cv2.putText(debug_tilt, f"{color}_BR", (int(br[0])+10, int(br[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
                cv2.line(debug_tilt, (int(bl[0]), int(bl[1])), (int(br[0]), int(br[1])), (0,255,0), 2)
                cv2.line(debug_tilt, (int(bl[0]), int(bl[1])), (int(bl[0] + square_width_px), int(bl[1])), (255,255,0), 1)
                
                text_y = int(bl[1]) + 30
                cv2.putText(debug_tilt, f"Angle: {angle_deg:.4f}°", (int(bl[0]), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(debug_tilt, f"Corr: {final_correction_um:.1f}μm", (int(bl[0]), text_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            self.progress.emit("Analysis completed!")
            
            # Calculate adjustment values
            adjustments = calculate_adjustment_values(results_reg, results_tilt)
            
            # Create additional visualization images
            cmyk_detection_img = self.create_cmyk_detection_image(cropped, HSV)
            p_points_img = self.create_p_points_image(cropped, results_reg, h_px)
            t_points_img = self.create_t_points_image(cropped, target_coords, h_px)
            
            # Return results
            results = {
                'registration': results_reg,
                'tilt': results_tilt,
                'adjustments': adjustments,  # LATERAL, CIRCUM, COCKING values
                'debug_reg': debug_reg,
                'debug_tilt': debug_tilt,
                'preprocessed': preprocessed,  # Preprocessed image
                'cmyk_detection': cmyk_detection_img,  # CMYK color boxes detected
                'p_points': p_points_img,  # P points (actual positions)
                't_points': t_points_img,  # T points (target positions)
                'metadata': {
                    'image_path': self.image_path,
                    'print_width_mm': self.print_width_mm,
                    'image_size_px': [w_px, h_px],
                    'mm_per_pixel': [mm_per_pixel_x, mm_per_pixel_y],
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"Error during analysis: {str(e)}")
            
    def create_cmyk_detection_image(self, cropped, HSV):
        """Create image showing CMYK color detection results"""
        result_img = cropped.copy()
        
        for color, hsv_range in HSV.items():
            # Create mask for this color
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
            
            # Find contours
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours - only keep large ones (main color squares)
            if cnts:
                # Calculate areas and find the largest one
                areas = [cv2.contourArea(c) for c in cnts]
                max_area = max(areas) if areas else 0
                
                # Only keep contours that are at least 30% of the largest area
                # This filters out small markers and keeps only main squares
                min_area_threshold = max_area * 0.3
                large_cnts = [c for c, area in zip(cnts, areas) if area >= min_area_threshold]
                
                # Draw only the large contours
                color_map = {'C': (255, 255, 0), 'M': (255, 0, 255), 'Y': (0, 255, 255), 'K': (128, 128, 128)}
                draw_color = color_map.get(color, (255, 255, 255))
                
                cv2.drawContours(result_img, large_cnts, -1, draw_color, 3)
                
                # Add color label on the largest contour only
                if large_cnts:
                    largest_cnt = max(large_cnts, key=cv2.contourArea)
                    M = cv2.moments(largest_cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(result_img, f"{color}", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, draw_color, 3)
        
        return result_img
    
    def create_p_points_image(self, cropped, results_reg, h_px):
        """Create image showing P points (actual detected positions)"""
        result_img = cropped.copy()
        
        for color, reg_data in results_reg.items():
            if reg_data is None:
                continue
                
            # Get P coordinate
            px_mm, py_mm = reg_data['P_coord_mm']
            
            # Convert back to pixel coordinates
            mm_per_pixel_x = 5.0 / cropped.shape[1]
            mm_per_pixel_y = 5.0 / cropped.shape[0]
            
            px_bl = px_mm / mm_per_pixel_x
            py_bl = py_mm / mm_per_pixel_y
            
            # Convert to OpenCV coordinates
            px_cv = int(px_bl)
            py_cv = int(h_px - py_bl)
            
            # Draw P point
            color_map = {'C': (255, 255, 0), 'M': (255, 0, 255), 'Y': (0, 255, 255), 'K': (255, 255, 255)}
            draw_color = color_map.get(color, (255, 255, 255))
            
            cv2.circle(result_img, (px_cv, py_cv), 12, draw_color, -1)
            cv2.circle(result_img, (px_cv, py_cv), 15, (0, 0, 0), 3)
            cv2.putText(result_img, f"P{color}", (px_cv+20, py_cv), cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)
            cv2.putText(result_img, f"({px_mm:.2f},{py_mm:.2f})", (px_cv+20, py_cv+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_img
    
    def create_t_points_image(self, cropped, target_coords, h_px):
        """Create image showing T points (target positions)"""
        result_img = cropped.copy()
        
        for color, (tx_px, ty_px) in target_coords.items():
            tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
            tx_cv = int(tx_bl)
            ty_cv = int(h_px - ty_bl)
            
            # Draw T point
            color_map = {'C': (255, 255, 0), 'M': (255, 0, 255), 'Y': (0, 255, 255), 'K': (255, 255, 255)}
            draw_color = color_map.get(color, (255, 255, 255))
            
            # Draw X mark
            cv2.line(result_img, (tx_cv-15, ty_cv-15), (tx_cv+15, ty_cv+15), draw_color, 4)
            cv2.line(result_img, (tx_cv-15, ty_cv+15), (tx_cv+15, ty_cv-15), draw_color, 4)
            cv2.line(result_img, (tx_cv-15, ty_cv-15), (tx_cv+15, ty_cv+15), (0, 0, 0), 2)
            cv2.line(result_img, (tx_cv-15, ty_cv+15), (tx_cv+15, ty_cv-15), (0, 0, 0), 2)
            
            cv2.putText(result_img, f"T{color}", (tx_cv+20, ty_cv), cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)
            
            # Convert to mm for display
            mm_per_pixel_x = 5.0 / cropped.shape[1]
            mm_per_pixel_y = 5.0 / cropped.shape[0]
            tx_mm = tx_bl * mm_per_pixel_x
            ty_mm = ty_bl * mm_per_pixel_y
            cv2.putText(result_img, f"({tx_mm:.2f},{ty_mm:.2f})", (tx_cv+20, ty_cv+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_img

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
                # OpenCV BGR을 RGB로 변환
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

class CMYKAnalyzerGUI(QMainWindow):
    """Main GUI application"""
    
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.analysis_results = None
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("CMYK Registration & Tilt Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel (image input and settings)
        left_panel = self.create_left_panel()
        
        # Right panel (results display)
        right_panel = self.create_right_panel()
        
        # Split with splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 900])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_left_panel(self):
        """Create left panel (image input and settings)"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Image input group
        input_group = QGroupBox("📸 Image Input")
        input_layout = QVBoxLayout()
        
        # Image upload button
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 12px; }")
        input_layout.addWidget(self.upload_btn)
        
        # Camera capture widget
        self.camera_widget = CameraCaptureWidget()
        self.camera_widget.image_captured.connect(self.auto_analyze_captured_image)
        input_layout.addWidget(self.camera_widget)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Settings group
        settings_group = QGroupBox("⚙️ Analysis Settings")
        settings_layout = QGridLayout()
        
        # Print width setting
        settings_layout.addWidget(QLabel("Print Width (mm):"), 0, 0)
        self.width_spinbox = QDoubleSpinBox()
        self.width_spinbox.setRange(1.0, 1000.0)
        self.width_spinbox.setValue(210.0)  # A4 default
        self.width_spinbox.setSuffix(" mm")
        settings_layout.addWidget(self.width_spinbox, 0, 1)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel("Upload image or capture to auto-analyze\n💡 Tip: Use images with clearly visible square markers")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def create_right_panel(self):
        """Create right panel (results display)"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Result tab widget
        self.result_tabs = QTabWidget()
        self.result_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                top: -1px;
            }
            QTabBar::tab {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f8f9fa, stop: 1 #e9ecef);
                border: 1px solid #dee2e6;
                padding: 8px 12px;
                margin-right: 1px;
                font-size: 12px;
                font-weight: bold;
                min-width: 80px;
                min-height: 20px;
                color: #495057;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #007bff, stop: 1 #0056b3);
                border-bottom-color: #007bff;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #e2e6ea, stop: 1 #dae0e5);
            }
        """)
        
        # 1. Overview tab
        self.overview_tab = QWidget()
        self.overview_layout = QVBoxLayout()
        
        # Scroll area for overview cards
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.overview_cards_layout = QVBoxLayout()
        scroll_widget.setLayout(self.overview_cards_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.overview_layout.addWidget(scroll_area)
        self.overview_tab.setLayout(self.overview_layout)
        self.result_tabs.addTab(self.overview_tab, "Analysis Overview")
        
        # 2. Registration tab
        self.registration_tab = QWidget()
        self.registration_layout = QVBoxLayout()
        self.registration_text = QTextEdit()
        self.registration_text.setReadOnly(True)
        # Set larger font for better readability
        font = QFont("Arial", 12)
        self.registration_text.setFont(font)
        self.registration_layout.addWidget(self.registration_text)
        self.registration_tab.setLayout(self.registration_layout)
        self.result_tabs.addTab(self.registration_tab, "Color Registration")
        
        # 3. Tilt analysis tab
        self.tilt_tab = QWidget()
        self.tilt_layout = QVBoxLayout()
        self.tilt_text = QTextEdit()
        self.tilt_text.setReadOnly(True)
        # Set larger font for better readability
        font = QFont("Arial", 12)
        self.tilt_text.setFont(font)
        self.tilt_layout.addWidget(self.tilt_text)
        self.tilt_tab.setLayout(self.tilt_layout)
        self.result_tabs.addTab(self.tilt_tab, "Tilt Analysis")
        
        # 4. Adjustment values tab (실무용)
        self.adjustment_tab = QWidget()
        self.adjustment_layout = QVBoxLayout()
        
        # Title label
        title_label = QLabel("🔧 PRINTING MACHINE ADJUSTMENT VALUES")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        self.adjustment_layout.addWidget(title_label)
        
        # Instructions label
        instructions_label = QLabel("Use these values to adjust your printing machine:")
        instructions_label.setStyleSheet("font-size: 14px; color: #34495e; margin: 5px;")
        instructions_label.setAlignment(Qt.AlignCenter)
        self.adjustment_layout.addWidget(instructions_label)
        
        # Adjustment table
        self.adjustment_table = QTableWidget()
        self.adjustment_table.setRowCount(4)  # C, M, Y, K
        self.adjustment_table.setColumnCount(4)  # LATERAL, CIRCUM, COCKING, Status
        self.adjustment_table.setHorizontalHeaderLabels(['LATERAL (mm)', 'CIRCUM (mm)', 'COCKING (mm)', 'Status'])
        
        # Set row labels
        colors = ['C (Cyan)', 'M (Magenta)', 'Y (Yellow)', 'K (Black)']
        self.adjustment_table.setVerticalHeaderLabels(colors)
        
        # Style the table
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
        
        # Set table properties
        self.adjustment_table.setAlternatingRowColors(True)
        self.adjustment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.adjustment_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.adjustment_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.adjustment_layout.addWidget(self.adjustment_table)
        
        # Instructions group
        instructions_group = QGroupBox("📋 Adjustment Instructions")
        instructions_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        instructions_layout = QVBoxLayout()
        
        instructions_text = """
• LATERAL: Adjust left-right position
  - Positive value: Move RIGHT
  - Negative value: Move LEFT

• CIRCUM: Adjust up-down position
  - Positive value: Move UP
  - Negative value: Move DOWN

• COCKING: Adjust rotation around a point
  - Positive value: Rotate CLOCKWISE (up)
  - Negative value: Rotate COUNTER-CLOCKWISE (down)

💡 TIP: Make small adjustments and re-analyze for best results.
        """
        
        instructions_detail = QLabel(instructions_text.strip())
        instructions_detail.setStyleSheet("font-size: 12px; color: #2c3e50; margin: 10px;")
        instructions_detail.setWordWrap(True)
        instructions_layout.addWidget(instructions_detail)
        instructions_group.setLayout(instructions_layout)
        
        self.adjustment_layout.addWidget(instructions_group)
        
        self.adjustment_tab.setLayout(self.adjustment_layout)
        self.result_tabs.addTab(self.adjustment_tab, "Adjustment Values")
        
        # 5. Visualization tab
        self.visualization_tab = QWidget()
        self.visualization_layout = QVBoxLayout()
        
        # Title
        viz_title = QLabel("🖼️ Analysis Visualization")
        viz_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 10px;")
        viz_title.setAlignment(Qt.AlignCenter)
        self.visualization_layout.addWidget(viz_title)
        
        # Image selection buttons (5 images)
        image_btn_layout = QGridLayout()
        
        self.show_preprocessed_btn = QPushButton("1. Preprocessed Image\n(Square Extraction)")
        self.show_preprocessed_btn.clicked.connect(lambda: self.show_debug_image('preprocessed'))
        self.show_preprocessed_btn.setStyleSheet("QPushButton { padding: 15px; font-size: 16px; font-weight: bold; }")
        
        self.show_cmyk_btn = QPushButton("2. CMYK Detection\n(Color Boxes Found)")
        self.show_cmyk_btn.clicked.connect(lambda: self.show_debug_image('cmyk_detection'))
        self.show_cmyk_btn.setStyleSheet("QPushButton { padding: 15px; font-size: 16px; font-weight: bold; }")
        
        self.show_p_points_btn = QPushButton("3. P Points\n(Bottom-Left Corners)")
        self.show_p_points_btn.clicked.connect(lambda: self.show_debug_image('p_points'))
        self.show_p_points_btn.setStyleSheet("QPushButton { padding: 15px; font-size: 16px; font-weight: bold; }")
        
        self.show_t_points_btn = QPushButton("4. T Points\n(Target Positions)")
        self.show_t_points_btn.clicked.connect(lambda: self.show_debug_image('t_points'))
        self.show_t_points_btn.setStyleSheet("QPushButton { padding: 15px; font-size: 16px; font-weight: bold; }")
        
        self.show_tilt_btn = QPushButton("5. Tilt Analysis\n(Angle & Correction)")
        self.show_tilt_btn.clicked.connect(lambda: self.show_debug_image('tilt'))
        self.show_tilt_btn.setStyleSheet("QPushButton { padding: 15px; font-size: 16px; font-weight: bold; }")
        
        # Arrange buttons in grid (3 columns, 2 rows)
        image_btn_layout.addWidget(self.show_preprocessed_btn, 0, 0)
        image_btn_layout.addWidget(self.show_cmyk_btn, 0, 1)
        image_btn_layout.addWidget(self.show_p_points_btn, 0, 2)
        image_btn_layout.addWidget(self.show_t_points_btn, 1, 0)
        image_btn_layout.addWidget(self.show_tilt_btn, 1, 1)
        
        # Add empty widget to balance the grid
        empty_widget = QWidget()
        image_btn_layout.addWidget(empty_widget, 1, 2)
        
        self.visualization_layout.addLayout(image_btn_layout)
        
        # Image display area
        self.image_scroll = QScrollArea()
        self.image_label = QLabel("Select an image type above to view analysis results")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9; font-size: 14px; color: #7f8c8d;")
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(True)
        self.visualization_layout.addWidget(self.image_scroll)
        
        self.visualization_tab.setLayout(self.visualization_layout)
        self.result_tabs.addTab(self.visualization_tab, "Visualization")
        
        layout.addWidget(self.result_tabs)
        
        # Save buttons
        save_layout = QHBoxLayout()
        self.save_json_btn = QPushButton("💾 Save JSON")
        self.save_json_btn.clicked.connect(self.save_json)
        self.save_json_btn.setEnabled(False)
        self.save_json_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 12px; }")
        
        self.save_image_btn = QPushButton("🖼️ Save Image")
        self.save_image_btn.clicked.connect(self.save_debug_image)
        self.save_image_btn.setEnabled(False)
        self.save_image_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 12px; }")
        
        save_layout.addWidget(self.save_json_btn)
        save_layout.addWidget(self.save_image_btn)
        layout.addLayout(save_layout)
        
        panel.setLayout(layout)
        return panel
        
    def upload_image(self):
        """Upload image and auto-analyze"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.load_image(file_path)
            self.start_analysis()
            
    def auto_analyze_captured_image(self, image_path):
        """Automatically analyze captured image"""
        print(f"DEBUG: Auto-analyzing captured image: {image_path}")  # Debug message
        self.load_image(image_path)
        self.start_analysis()
        
    def load_captured_image(self, image_path):
        """Load captured image"""
        self.load_image(image_path)
        
    def load_image(self, image_path):
        """Load image and update status"""
        self.current_image_path = image_path
        self.status_label.setText(f"Loaded image: {os.path.basename(image_path)}")
        self.statusBar().showMessage(f"Image loaded: {os.path.basename(image_path)}")
        print(f"DEBUG: Image loaded: {image_path}")  # Debug message
        
    def start_analysis(self):
        """Start analysis"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return
            
        print(f"DEBUG: Starting analysis for: {self.current_image_path}")  # Debug message
        
        # Stop any existing worker
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            
        # Update UI state
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Infinite progress
        self.status_label.setText("Analyzing...")
        
        # Start worker thread
        self.worker = AnalysisWorker(self.current_image_path, self.width_spinbox.value())
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.start()
        
    def update_progress(self, message):
        """Update progress"""
        self.status_label.setText(message)
        self.statusBar().showMessage(message)
        
    def analysis_finished(self, results):
        """Analysis completed"""
        self.analysis_results = results
        print("DEBUG: Analysis completed successfully")  # Debug message
        
        # Clean up worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        # Restore UI state
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis completed!")
        
        # Display results
        self.display_results(results)
        
        # Enable save buttons
        self.save_json_btn.setEnabled(True)
        self.save_image_btn.setEnabled(True)
        
        self.statusBar().showMessage("Analysis completed")
        
    def analysis_error(self, error_message):
        """Analysis error"""
        print(f"DEBUG: Analysis error: {error_message}")  # Debug message
        
        # Clean up worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.statusBar().showMessage("Analysis failed")
        
    def display_results(self, results):
        """Display results in each tab"""
        # 1. Overview tab
        self.populate_overview_cards(results)
        
        # 2. Registration tab
        registration_text = self.generate_registration_text(results['registration'])
        self.registration_text.setPlainText(registration_text)
        
        # 3. Tilt analysis tab
        tilt_text = self.generate_tilt_text(results['tilt'])
        self.tilt_text.setPlainText(tilt_text)
        
        # 4. Adjustment values tab
        self.populate_adjustment_table(results['adjustments'])
        
        # 5. Visualization tab - show first image by default
        self.show_debug_image('preprocessed')
        
        # Force complete UI refresh for all tabs
        self.force_ui_refresh()
        
        # Switch to adjustment values tab (most important for field use)
        self.result_tabs.setCurrentIndex(3)  # Index 3 is adjustment values tab
        
    def force_ui_refresh(self):
        """Force complete UI refresh for all tabs and widgets"""
        print("DEBUG: Forcing complete UI refresh...")
        
        # Refresh all tabs
        for i in range(self.result_tabs.count()):
            widget = self.result_tabs.widget(i)
            if widget:
                widget.update()
                widget.repaint()
        
        # Refresh specific widgets
        widgets_to_refresh = [
            self.adjustment_table,
            self.overview_scroll_area,
            self.vis_image_label,
            self.status_bar
        ]
        
        for widget in widgets_to_refresh:
            if widget:
                widget.update()
                widget.repaint()
        
        # Force overall refresh
        self.result_tabs.update()
        self.result_tabs.repaint()
        
        # Process all pending events
        QApplication.processEvents()
        
        print("DEBUG: UI refresh completed")

    def generate_overview_text(self, results):
        """Generate overview text"""
        text = "=== CMYK Registration & Tilt Analysis Report ===\n\n"
        
        # Metadata
        meta = results['metadata']
        text += f"📅 Analysis Time: {meta['timestamp']}\n"
        text += f"📁 Image File: {os.path.basename(meta['image_path'])}\n"
        text += f"📏 Print Width: {meta['print_width_mm']} mm\n"
        text += f"🖼️ Image Size: {meta['image_size_px'][0]} x {meta['image_size_px'][1]} pixels\n"
        text += f"📐 mm per pixel: {meta['mm_per_pixel'][0]:.6f} x {meta['mm_per_pixel'][1]:.6f}\n"
        text += f"✅ Preprocessing: Square marker extraction completed\n\n"
        
        # Color-wise summary
        text += "=== Color-wise Analysis Summary ===\n\n"
        
        reg_results = results['registration']
        tilt_results = results['tilt']
        
        for color in ['C', 'M', 'Y', 'K']:
            text += f"🎨 {color} (Cyan/Magenta/Yellow/Black):\n"
            
            # Registration results
            if reg_results.get(color):
                reg = reg_results[color]
                dx, dy = reg['movement_mm']
                text += f"  📍 Registration: ({dx:+.3f}, {dy:+.3f}) mm\n"
            else:
                text += f"  ❌ Registration: Detection failed\n"
                
            # Tilt results
            if tilt_results.get(color):
                tilt = tilt_results[color]
                angle = tilt['tilt_angle_deg']
                corr = tilt['correction_actual_um']
                corr_mm = corr / 1000.0  # Convert to mm for display
                text += f"  📐 Tilt: {angle:+.6f}° (Correction: {corr_mm:+.3f} mm)\n"
            else:
                text += f"  ❌ Tilt: Detection failed\n"
                
            text += "\n"
            
        return text
        
    def generate_adjustment_text(self, adjustments):
        """Generate practical adjustment values text for field use"""
        text = "=== PRINTING MACHINE ADJUSTMENT VALUES ===\n\n"
        text += "🔧 Use these values to adjust your printing machine:\n\n"
        
        # Create a more readable table format
        text += "┌─────────┬──────────────┬─────────────┬──────────────┬────────┐\n"
        text += "│ Color   │ LATERAL (mm) │ CIRCUM (mm) │ COCKING (mm) │ Status │\n"
        text += "├─────────┼──────────────┼─────────────┼──────────────┼────────┤\n"
        
        for color in ['C', 'M', 'Y', 'K']:
            adj = adjustments.get(color, {})
            
            if adj.get('status') == 'OK':
                lateral = f"{adj['LATERAL']:+.3f}" if adj['LATERAL'] is not None else "N/A"
                circum = f"{adj['CIRCUM']:+.3f}" if adj['CIRCUM'] is not None else "N/A"
                cocking = f"{adj['COCKING']:+.3f}" if adj['COCKING'] is not None else "N/A"
                status = "✅ OK"
            else:
                lateral = "N/A"
                circum = "N/A"
                cocking = "N/A"
                status = "❌ Failed"
            
            text += f"│ {color:7} │ {lateral:12} │ {circum:11} │ {cocking:12} │ {status:6} │\n"
        
        text += "└─────────┴──────────────┴─────────────┴──────────────┴────────┘\n"
        
        text += "\n" + "="*80 + "\n"
        text += "📋 ADJUSTMENT INSTRUCTIONS:\n\n"
        text += "• LATERAL: Adjust left-right position\n"
        text += "  - Positive value: Move RIGHT\n"
        text += "  - Negative value: Move LEFT\n\n"
        text += "• CIRCUM: Adjust up-down position\n"
        text += "  - Positive value: Move UP\n"
        text += "  - Negative value: Move DOWN\n\n"
        text += "• COCKING: Adjust rotation around a point\n"
        text += "  - Positive value: Rotate CLOCKWISE (up)\n"
        text += "  - Negative value: Rotate COUNTER-CLOCKWISE (down)\n\n"
        text += "💡 TIP: Make small adjustments and re-analyze for best results.\n"
        
        return text
        
    def populate_adjustment_table(self, adjustments):
        """Populate the adjustment table with results"""
        colors = ['C', 'M', 'Y', 'K']
        
        # Clear existing items first
        self.adjustment_table.clearContents()
        
        for row, color in enumerate(colors):
            adj = adjustments.get(color, {})
            
            if adj.get('status') == 'OK':
                # LATERAL
                lateral_item = QTableWidgetItem(f"{adj['LATERAL']:+.3f}" if adj['LATERAL'] is not None else "N/A")
                lateral_item.setTextAlignment(Qt.AlignCenter)
                if adj['LATERAL'] is not None:
                    if adj['LATERAL'] > 0:
                        lateral_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                    else:
                        lateral_item.setBackground(QColor(235, 255, 235))  # Light green for negative
                
                # CIRCUM  
                circum_item = QTableWidgetItem(f"{adj['CIRCUM']:+.3f}" if adj['CIRCUM'] is not None else "N/A")
                circum_item.setTextAlignment(Qt.AlignCenter)
                if adj['CIRCUM'] is not None:
                    if adj['CIRCUM'] > 0:
                        circum_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                    else:
                        circum_item.setBackground(QColor(235, 255, 235))  # Light green for negative
                
                # COCKING
                cocking_item = QTableWidgetItem(f"{adj['COCKING']:+.3f}" if adj['COCKING'] is not None else "N/A")
                cocking_item.setTextAlignment(Qt.AlignCenter)
                if adj['COCKING'] is not None:
                    if adj['COCKING'] > 0:
                        cocking_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                    else:
                        cocking_item.setBackground(QColor(235, 255, 235))  # Light green for negative
                
                # Status
                status_item = QTableWidgetItem("✅ OK")
                status_item.setTextAlignment(Qt.AlignCenter)
                status_item.setBackground(QColor(235, 255, 235))  # Light green
                
            else:
                # Failed detection
                lateral_item = QTableWidgetItem("N/A")
                circum_item = QTableWidgetItem("N/A") 
                cocking_item = QTableWidgetItem("N/A")
                status_item = QTableWidgetItem("❌ Failed")
                
                # Set gray background for failed items
                gray_color = QColor(240, 240, 240)
                lateral_item.setBackground(gray_color)
                circum_item.setBackground(gray_color)
                cocking_item.setBackground(gray_color)
                status_item.setBackground(QColor(255, 220, 220))  # Light red
                
                # Center align
                lateral_item.setTextAlignment(Qt.AlignCenter)
                circum_item.setTextAlignment(Qt.AlignCenter)
                cocking_item.setTextAlignment(Qt.AlignCenter)
                status_item.setTextAlignment(Qt.AlignCenter)
            
            # Set items in table
            self.adjustment_table.setItem(row, 0, lateral_item)
            self.adjustment_table.setItem(row, 1, circum_item)
            self.adjustment_table.setItem(row, 2, cocking_item)
            self.adjustment_table.setItem(row, 3, status_item)
        
        # Resize table to fit contents
        self.adjustment_table.resizeRowsToContents()
        
    def populate_overview_cards(self, results):
        """Populate overview with cards"""
        print("DEBUG: Populating overview cards...")
        
        # Clear existing cards completely
        for i in reversed(range(self.overview_cards_layout.count())):
            item = self.overview_cards_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                self.overview_cards_layout.removeItem(item)
        
        # Metadata card
        meta = results['metadata']
        meta_card = self.create_info_card(
            "📋 Analysis Information",
            [
                f"📅 Analysis Time: {meta['timestamp'][:19].replace('T', ' ')}",
                f"📁 Image File: {os.path.basename(meta['image_path'])}",
                f"📏 Print Width: {meta['print_width_mm']} mm",
                f"🖼️ Image Size: {meta['image_size_px'][0]} x {meta['image_size_px'][1]} pixels",
                f"📐 Resolution: {meta['mm_per_pixel'][0]:.6f} x {meta['mm_per_pixel'][1]:.6f} mm/pixel",
                f"✅ Preprocessing: Square marker extraction completed"
            ]
        )
        self.overview_cards_layout.addWidget(meta_card)
        
        # Color summary cards
        reg_results = results['registration']
        tilt_results = results['tilt']
        adj_results = results['adjustments']
        
        # Create a grid layout for color cards
        colors_widget = QWidget()
        colors_layout = QGridLayout()
        
        color_names = {'C': 'Cyan', 'M': 'Magenta', 'Y': 'Yellow', 'K': 'Black'}
        color_emojis = {'C': '🔵', 'M': '🟣', 'Y': '🟡', 'K': '⚫'}
        
        for i, color in enumerate(['C', 'M', 'Y', 'K']):
            # Get data
            reg_data = reg_results.get(color)
            tilt_data = tilt_results.get(color)
            adj_data = adj_results.get(color)
            
            info_lines = []
            
            if reg_data:
                dx, dy = reg_data['movement_mm']
                info_lines.append(f"📍 Registration: ({dx:+.3f}, {dy:+.3f}) mm")
                info_lines.append(f"   → LATERAL: {dx:+.3f} mm ({'RIGHT' if dx > 0 else 'LEFT'})")
                info_lines.append(f"   → CIRCUM: {dy:+.3f} mm ({'UP' if dy > 0 else 'DOWN'})")
            else:
                info_lines.append("❌ Registration: Detection failed")
                
            if tilt_data:
                angle = tilt_data['tilt_angle_deg']
                corr = tilt_data['correction_actual_um']
                corr_mm = corr / 1000.0
                info_lines.append(f"📐 Tilt: {angle:+.6f}° ({corr_mm:+.3f} mm)")
            else:
                info_lines.append("❌ Tilt: Detection failed")
                
            if adj_data and adj_data.get('status') == 'OK':
                lateral = adj_data['LATERAL']
                circum = adj_data['CIRCUM']
                cocking = adj_data['COCKING']
                info_lines.append(f"🔧 Adjustments:")
                info_lines.append(f"   LATERAL: {lateral:+.3f} mm")
                info_lines.append(f"   CIRCUM: {circum:+.3f} mm")
                info_lines.append(f"   COCKING: {cocking:+.3f} mm")
            else:
                info_lines.append("❌ Adjustments: Cannot calculate")
            
            # Create card
            card = self.create_info_card(
                f"{color_emojis[color]} {color} ({color_names[color]})",
                info_lines
            )
            
            # Add status styling
            if adj_data and adj_data.get('status') == 'OK':
                card.setStyleSheet(card.styleSheet() + "QGroupBox { border-left: 5px solid #27ae60; }")
            else:
                card.setStyleSheet(card.styleSheet() + "QGroupBox { border-left: 5px solid #e74c3c; }")
            
            colors_layout.addWidget(card, i // 2, i % 2)
        
        colors_widget.setLayout(colors_layout)
        self.overview_cards_layout.addWidget(colors_widget)
        
        # Add stretch to push cards to top
        self.overview_cards_layout.addStretch()
        
    def create_info_card(self, title, info_lines):
        """Create an information card widget"""
        card = QGroupBox(title)
        card.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
                background-color: white;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout()
        
        for line in info_lines:
            label = QLabel(line)
            label.setStyleSheet("font-size: 12px; font-weight: normal; color: #34495e; margin: 2px 0px;")
            label.setWordWrap(True)
            layout.addWidget(label)
        
        card.setLayout(layout)
        return card
        
    def generate_registration_text(self, reg_results):
        """Generate registration results text"""
        text = "=== Color Registration Detailed Analysis ===\n\n"
        
        for color in ['C', 'M', 'Y', 'K']:
            text += f"🎨 {color} Color:\n"
            
            if reg_results.get(color):
                reg = reg_results[color]
                px, py = reg['P_coord_mm']
                tx, ty = reg['T_coord_mm']
                dx, dy = reg['movement_mm']
                
                text += f"  📍 Actual Position (P): ({px:.3f}, {py:.3f}) mm\n"
                text += f"  🎯 Target Position (T): ({tx:.3f}, {ty:.3f}) mm\n"
                text += f"  ➡️  Movement: ({dx:+.3f}, {dy:+.3f}) mm\n"
                
                # Movement direction description
                if abs(dx) > 0.001:
                    direction_x = "right" if dx > 0 else "left"
                    text += f"     X-axis: {abs(dx):.3f}mm move {direction_x}\n"
                if abs(dy) > 0.001:
                    direction_y = "up" if dy > 0 else "down"
                    text += f"     Y-axis: {abs(dy):.3f}mm move {direction_y}\n"
            else:
                text += f"  ❌ Detection failed\n"
                
            text += "\n"
            
        return text
        
    def generate_tilt_text(self, tilt_results):
        """Generate tilt analysis results text"""
        text = "=== Tilt Analysis Detailed Results ===\n\n"
        
        for color in ['C', 'M', 'Y', 'K']:
            text += f"🎨 {color} Color:\n"
            
            if tilt_results.get(color):
                tilt = tilt_results[color]
                
                bl_px = tilt['bottom_left_px']
                br_px = tilt['bottom_right_px']
                angle = tilt['tilt_angle_deg']
                corr_5mm = tilt['correction_5mm_um']
                corr_actual = tilt['correction_actual_um']
                
                text += f"  📍 Bottom Left Corner: ({bl_px[0]:.1f}, {bl_px[1]:.1f}) px\n"
                text += f"  📍 Bottom Right Corner: ({br_px[0]:.1f}, {br_px[1]:.1f}) px\n"
                text += f"  📐 Tilt Angle: {angle:+.6f}°\n"
                text += f"  🔧 5mm Reference Correction: {corr_5mm:+.3f} μm\n"
                text += f"  🔧 Actual Correction: {corr_actual:+.3f} μm\n"
                
                # Correction direction description
                if abs(corr_actual) > 0.1:
                    direction = "up" if corr_actual > 0 else "down"
                    text += f"  ➡️  Move bottom right corner {abs(corr_actual):.1f}μm {direction}\n"
                else:
                    text += f"  ✅ No tilt correction needed (error < 0.1μm)\n"
            else:
                text += f"  ❌ Detection failed\n"
                
            text += "\n"
            
        return text
        
    def show_debug_image(self, image_type):
        """Display debug image"""
        print(f"DEBUG: Showing debug image: {image_type}")
        
        if not self.analysis_results:
            self.vis_image_label.setText("No analysis results available")
            self.vis_image_label.update()
            return
            
        # Get the appropriate image based on type
        if image_type == 'preprocessed':
            debug_img = self.analysis_results['preprocessed']
        elif image_type == 'cmyk_detection':
            debug_img = self.analysis_results['cmyk_detection']
        elif image_type == 'p_points':
            debug_img = self.analysis_results['p_points']
        elif image_type == 't_points':
            debug_img = self.analysis_results['t_points']
        elif image_type == 'tilt':
            debug_img = self.analysis_results['debug_tilt']
        elif image_type == 'registration':
            debug_img = self.analysis_results['debug_reg']
        else:
            return
            
        # Convert OpenCV BGR to RGB
        rgb_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit label size while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        
        # Update label text based on image type
        image_descriptions = {
            'preprocessed': "1. Preprocessed Image - Square marker extracted and perspective corrected",
            'cmyk_detection': "2. CMYK Detection - Color boxes found and outlined with their labels",
            'p_points': "3. P Points - Actual detected bottom-left corners of color boxes",
            't_points': "4. T Points - Target positions where color boxes should be located",
            'tilt': "5. Tilt Analysis - Bottom edge angles and correction values"
        }
        
        # You can add a status label or tooltip here if needed
        description = image_descriptions.get(image_type, f"Analysis result: {image_type}")
        self.image_label.setToolTip(description)
        
        # Force immediate image update
        self.image_label.update()
        self.image_label.repaint()
        print(f"DEBUG: Image {image_type} displayed and updated")
        
    def save_json(self):
        """Save JSON results"""
        if not self.analysis_results:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON", f"cmyk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                # Create a copy of results without image data for JSON serialization
                json_results = {}
                for key, value in self.analysis_results.items():
                    if key in ['debug_reg', 'debug_tilt', 'preprocessed', 'cmyk_detection', 'p_points', 't_points']:
                        # Skip image data - cannot be JSON serialized
                        continue
                    json_results[key] = value
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_results, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "Save Complete", f"Results saved:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error occurred during save:\n{str(e)}")
                
    def save_debug_image(self):
        """Save debug image"""
        if not self.analysis_results:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", f"cmyk_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Images (*.png)"
        )
        
        if file_path:
            try:
                # Check currently displayed image type
                current_pixmap = self.image_label.pixmap()
                if current_pixmap:
                    current_pixmap.save(file_path)
                    QMessageBox.information(self, "Save Complete", f"Image saved:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error occurred during save:\n{str(e)}")
                
    def closeEvent(self, event):
        """Cleanup when application closes"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        if hasattr(self, 'camera_widget'):
            self.camera_widget.stop_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set font - increased base font size
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