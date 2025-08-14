#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMYK Registration & Tilt Analyzer GUI
GUI application for industrial printing quality management and calibration through CMYK color box alignment and tilt analysis
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

# Import existing analysis modules
from color_registration_analysis import (
    extract_marker, detect_bottom_left, detect_square_corners,
    pixel_to_bottom_left_coord, order_points
)

def detect_special_color(img: np.ndarray, exclude_colors: dict) -> tuple:
    """
    New logic: First detect all box shapes, then find special color box
    Identify special color by checking if C, M, Y, S boxes are all the same size
    
    Args:
        img: Input image (BGR)
        exclude_colors: Dictionary of HSV ranges for colors to exclude (C, M, Y)
    
    Returns:
        tuple: (Detected color HSV range, color name) or (None, None)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    
    # Step 1: Detect all box shapes
    all_boxes = []
    
    # Find known boxes with C, M, Y colors
    known_boxes = {}
    for color_name, hsv_range in exclude_colors.items():
        mask = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
        
        # Remove noise using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area condition
                # Rectangle approximation
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # If it's a rectangle
                    # Store box information
                    box_info = {
                        'contour': contour,
                        'area': area,
                        'approx': approx,
                        'color': color_name,
                        'center': tuple(map(int, cv2.minEnclosingCircle(contour)[0]))
                    }
                    known_boxes[color_name] = box_info
                    all_boxes.append(box_info)
    
    if len(known_boxes) < 3:  # Need at least 3 out of C, M, Y
        print("⚠️ Could not find enough C, M, Y boxes.")
        return None, None
    
    # Step 2: Calculate average size of known boxes
    known_areas = [box['area'] for box in known_boxes.values()]
    avg_area = np.mean(known_areas)
    area_tolerance = avg_area * 0.3  # 30% tolerance
    
    print(f"📏 Average area of known boxes: {avg_area:.0f} (Tolerance: ±{area_tolerance:.0f})")
    
    # Step 3: Find boxes of similar size to known boxes in the entire image
    # Create mask for colors to exclude
    exclude_mask = np.zeros((h, w), dtype=np.uint8)
    for color_name, hsv_range in exclude_colors.items():
        mask = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
        exclude_mask = cv2.bitwise_or(exclude_mask, mask)
    
    # Find boxes in areas excluding the excluded colors
    remaining_mask = cv2.bitwise_not(exclude_mask)
    
    # Test various color ranges to find boxes
    color_ranges = [
        # Red series
        ((0, 50, 50), (10, 255, 255)),
        ((170, 50, 50), (180, 255, 255)),
        # Orange series
        ((10, 50, 50), (25, 255, 255)),
        # Green series
        ((35, 50, 50), (85, 255, 255)),
        # Blue series
        ((100, 50, 50), (130, 255, 255)),
        # Purple series
        ((130, 50, 50), (170, 255, 255)),
        # Pink series
        ((140, 30, 50), (170, 255, 255)),
        # Brown series
        ((10, 100, 20), (20, 255, 200)),
        # Gray series
        ((0, 0, 50), (180, 30, 200)),
    ]
    
    potential_special_boxes = []
    
    for lower, upper in color_ranges:
        # Search only in areas excluding excluded colors
        test_mask = cv2.bitwise_and(remaining_mask, 
                                   cv2.inRange(hsv, np.array(lower), np.array(upper)))
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        test_mask = cv2.morphologyEx(test_mask, cv2.MORPH_CLOSE, kernel)
        test_mask = cv2.morphologyEx(test_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check if size is similar to known boxes
            if abs(area - avg_area) <= area_tolerance:
                # Rectangle approximation
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # If it's a rectangle
                    # Calculate box center point
                    center = tuple(map(int, cv2.minEnclosingCircle(contour)[0]))
                    
                    # Check if too close to already known boxes
                    too_close = False
                    for known_box in known_boxes.values():
                        known_center = known_box['center']
                        distance = np.sqrt((center[0] - known_center[0])**2 + 
                                        (center[1] - known_center[1])**2)
                        if distance < 50:  # Too close if within 50 pixels
                            too_close = True
                            break
                    
                    if not too_close:
                        potential_special_boxes.append({
                            'contour': contour,
                            'area': area,
                            'approx': approx,
                            'center': center,
                            'hsv_range': (lower, upper)
                        })
    
    # Step 4: Select the most suitable special color box
    if not potential_special_boxes:
        print("❌ Could not find suitable special color box.")
        return None, None
    
    # Select box with most similar area
    best_box = min(potential_special_boxes, 
                   key=lambda x: abs(x['area'] - avg_area))
    
    print(f"🎯 Special color box found: Area {best_box['area']:.0f}, Center {best_box['center']}")
    
    # Step 5: Return HSV range of selected box
    hsv_lower, hsv_upper = best_box['hsv_range']
    
    # Determine color name
    h_center = (hsv_lower[0] + hsv_upper[0]) // 2
    if h_center < 15 or h_center > 165:
        color_name = "Red"
    elif h_center < 45:
        color_name = "Orange"
    elif h_center < 75:
        color_name = "Yellow"
    elif h_center < 105:
        color_name = "Green"
    elif h_center < 135:
        color_name = "Cyan"
    elif h_center < 165:
        color_name = "Blue"
    else:
        color_name = "Purple"
    
    return (hsv_lower, hsv_upper), f"Special_{color_name}"

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

def calculate_adjustment_values(registration_results, cocking_results=None):
    """
    Calculate LATERAL, CIRCUM, COCKING adjustment values for each color.
    
    LATERAL: Left-right movement (X-axis)
    CIRCUM: Up-down movement (Y-axis) 
    COCKING: Y coordinate difference between left and right images
    
    Args:
        registration_results: Color registration analysis results
        cocking_results: Cocking analysis results (optional)
        
    Returns:
        dict: Adjustment values for each color
    """
    adjustments = {}
    
    for color in ['C', 'M', 'Y', 'S']:
        # Get registration data
        reg_data = registration_results.get(color)
        cocking_data = cocking_results.get(color) if cocking_results else None
        
        if reg_data is None:
            adjustments[color] = {
                'LATERAL': None,
                'CIRCUM': None,
                'COCKING': None,
                'status': 'Detection failed'
            }
            continue
            
        # Extract movement values (in mm)
        dx_mm, dy_mm = reg_data['movement_mm']
        
        # Convert to practical adjustment values
        # LATERAL = X-axis movement (left-right)
        # Right = +, Left = -
        lateral_mm = dx_mm
        
        # CIRCUM = Y-axis movement (up-down)
        # Up = +, Down = -
        circum_mm = dy_mm
        
        # COCKING = Y coordinate difference (right - left)
        cocking_mm = cocking_data['cocking_mm'] if cocking_data else None
        
        adjustments[color] = {
            'LATERAL': round(lateral_mm, 3),
            'CIRCUM': round(circum_mm, 3),
            'COCKING': round(cocking_mm, 3) if cocking_mm is not None else None,
            'status': 'OK'
        }
    
    return adjustments

class AnalysisWorker(QThread):
    """Worker thread that executes analysis tasks in the background"""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, left_image_path, right_image_path=None):
        super().__init__()
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        
    def run(self):
        try:
            self.progress.emit("Loading image...")
            
            # Load left image
            left_orig = cv2.imread(self.left_image_path)
            if left_orig is None:
                self.error.emit("Cannot read left image file.")
                return
                
            self.progress.emit("Preprocessing square marker...")
            
            # Step 1: Square marker preprocessing (preprocess.ipynb logic)
            preprocessed = extract_robust_square_marker(left_orig)
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
            
            # Basic CMYK color ranges (excluding K)
            base_colors = {
                'C': ((90,80,80),(130,255,255)),   # Cyan
                'M': ((130,50,70),(170,255,255)),  # Magenta 
                'Y': ((15,60,60),(45,255,255)),    # Yellow - expanded range for better detection
            }
            
            # Detect special color
            self.progress.emit("Detecting special color...")
            special_hsv_range, special_color_name = detect_special_color(cropped, base_colors)
            
            if special_hsv_range is None:
                self.error.emit("Special color could not be detected. Please ensure the image contains a clear special color region.")
                return
            
            # Complete color ranges (including special color)
            HSV = base_colors.copy()
            HSV['S'] = special_hsv_range  # Mark special color as 'S'
            
            # Store special color name
            self.special_color_name = special_color_name
            
            # Target coordinates (bottom-left origin)
            target_coords = {
                'S': (w_px/10, h_px - h_px*6/10),  # Special color at K position
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
                bl = detect_bottom_left(cropped, hsv_range, min_area_ratio=0.7)  # More strict filtering
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
            
            self.progress.emit("Analysis completed!")
            
            # Calculate cocking values if right image is available
            cocking_results = {}
            cocking_debug_img = None
            
            if self.right_image_path:
                self.progress.emit("Analyzing right image for cocking calculation...")
                cocking_results, cocking_debug_img = self.calculate_cocking(HSV, cropped, results_reg, mm_per_pixel_x, mm_per_pixel_y)
            
            # Calculate adjustment values (now includes cocking)
            adjustments = calculate_adjustment_values(results_reg, cocking_results)
            
            # Create additional visualization images
            cmyk_detection_img = self.create_cmyk_detection_image(cropped, HSV)
            p_points_img = self.create_p_points_image(cropped, results_reg, h_px)
            t_points_img = self.create_t_points_image(cropped, target_coords, h_px)
            
            # Return results
            results = {
                'registration': results_reg,
                'cocking': cocking_results,  # Cocking values (Y coordinate differences)
                'adjustments': adjustments,  # LATERAL, CIRCUM, COCKING values
                'debug_reg': debug_reg,
                'cocking_debug': cocking_debug_img,  # Cocking visualization
                'preprocessed': preprocessed,  # Preprocessed image
                'cmyk_detection': cmyk_detection_img,  # CMYK color boxes detected
                'p_points': p_points_img,  # P points (actual positions)
                't_points': t_points_img,  # T points (target positions)
                'special_color_name': self.special_color_name,  # Special color name
                'metadata': {
                    'left_image_path': self.left_image_path,
                    'right_image_path': self.right_image_path,
                    'image_size_px': [w_px, h_px],
                    'mm_per_pixel': [mm_per_pixel_x, mm_per_pixel_y],
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"Error during analysis: {str(e)}")
            
    def calculate_cocking(self, HSV, left_cropped, left_results_reg, mm_per_pixel_x, mm_per_pixel_y):
        """Calculate cocking values by comparing Y coordinates between left and right images"""
        try:
            print(f"DEBUG: Calculating cocking with right image: {self.right_image_path}")
            print(f"DEBUG: Right image path type: {type(self.right_image_path)}")
            print(f"DEBUG: Right image path is None: {self.right_image_path is None}")
            print(f"DEBUG: Right image path is empty: {self.right_image_path == ''}")
            
            if not self.right_image_path:
                print("ERROR: Right image path is empty or None")
                return {}, None
                
            # Load and process right image
            right_orig = cv2.imread(self.right_image_path)
            if right_orig is None:
                print("Warning: Cannot read right image file")
                return {}, None
                
            # Process right image the same way as left
            print("DEBUG: Processing right image for square marker...")
            print(f"DEBUG: Right image shape: {right_orig.shape}")
            right_preprocessed = extract_robust_square_marker(right_orig)
            if right_preprocessed is None:
                print("ERROR: Cannot find square marker in right image")
                print("DEBUG: This usually means the image doesn't contain a clear square region")
                return {}, None
            else:
                print(f"DEBUG: Square marker found, preprocessed shape: {right_preprocessed.shape}")
                
            print("DEBUG: Extracting CMYK marker from right image...")
            right_cropped = extract_marker(right_preprocessed)
            if right_cropped is None:
                print("ERROR: Cannot find CMYK marker in right image")
                print("DEBUG: This usually means the image doesn't contain CMYK color boxes")
                return {}, None
            else:
                print(f"DEBUG: CMYK marker extracted, cropped shape: {right_cropped.shape}")
            
            # Analyze right image for Y coordinates
            right_results_reg = {}
            
            # Target coordinates (same as left)
            target_coords = {
                'C': (1.0, 3.0),  # Top-left
                'M': (3.0, 3.0),  # Top-right  
                'Y': (1.0, 1.0),  # Bottom-left
                'S': (3.0, 1.0)   # Bottom-right (special color)
            }
            
            # Get Y coordinates from right image
            print("DEBUG: Detecting colors in right image...")
            for color, hsv_range in HSV.items():
                print(f"DEBUG: Looking for {color} color in right image...")
                bl_point = detect_bottom_left(right_cropped, hsv_range, min_area_ratio=0.7)
                if bl_point is not None:
                    px, py = bl_point
                    # Create the same data structure as left image
                    right_results_reg[color] = {'bottom_left_px': [float(px), float(py)]}
                    print(f"DEBUG: Found {color} at ({px}, {py})")
                    print(f"DEBUG: Right data structure for {color}: {right_results_reg[color]}")
                else:
                    print(f"DEBUG: Could not find {color} in right image")
            
            # Calculate cocking results
            print(f"DEBUG: Left results: {list(left_results_reg.keys())}")
            print(f"DEBUG: Right results: {list(right_results_reg.keys())}")
            
            # Check if we have enough data for cocking calculation
            if len(right_results_reg) < 2:
                print(f"ERROR: Not enough colors detected in right image. Found: {list(right_results_reg.keys())}")
                print("DEBUG: Need at least 2 colors (C, M, Y, S) to calculate cocking")
                return {}, None
            
            # Convert left results to have bottom_left_px for cocking calculation
            left_results_for_cocking = {}
            for color, data in left_results_reg.items():
                if data and 'P_coord_mm' in data:
                    # Convert P_coord_mm to bottom_left_px format
                    px_mm, py_mm = data['P_coord_mm']
                    # Convert mm to pixels using mm_per_pixel
                    px_px = px_mm / mm_per_pixel_x
                    py_px = py_mm / mm_per_pixel_y
                    left_results_for_cocking[color] = {'bottom_left_px': [px_px, py_px]}
                    print(f"DEBUG: Converted {color} from mm ({px_mm}, {py_mm}) to px ({px_px}, {py_px})")
                else:
                    print(f"DEBUG: Skipping {color} - no P_coord_mm data")
            
            cocking_results = {}
            cocking_debug_img = self.create_cocking_debug_image(left_cropped, right_cropped, left_results_for_cocking, right_results_reg, mm_per_pixel_y)
            
            for color in ['C', 'M', 'Y', 'S']:
                left_data = left_results_for_cocking.get(color)
                right_data = right_results_reg.get(color)
                
                if left_data and right_data:
                    try:
                        # Get Y coordinates (bottom_left_px[1] is Y coordinate)
                        left_y_px = left_data['bottom_left_px'][1]
                        right_y_px = right_data['bottom_left_px'][1]
                        
                        # Calculate Y difference in pixels (right - left)
                        y_diff_px = right_y_px - left_y_px
                        
                        # Convert to mm
                        y_diff_mm = y_diff_px * mm_per_pixel_y
                        
                        cocking_results[color] = {
                            'left_y_px': left_y_px,
                            'right_y_px': right_y_px,
                            'y_diff_px': y_diff_px,
                            'cocking_mm': round(y_diff_mm, 3)
                        }
                        
                        print(f"Cocking {color}: Left Y={left_y_px:.1f}px, Right Y={right_y_px:.1f}px, Diff={y_diff_mm:.3f}mm")
                    except Exception as e:
                        print(f"ERROR: Failed to calculate cocking for {color}: {str(e)}")
                        cocking_results[color] = None
                else:
                    print(f"DEBUG: Skipping {color} - missing data")
                    cocking_results[color] = None
                    
            return cocking_results, cocking_debug_img
            
        except Exception as e:
            print(f"Error calculating cocking: {str(e)}")
            return {}, None
    
    def create_cocking_debug_image(self, left_cropped, right_cropped, left_results, right_results, mm_per_pixel_y):
        """Create debug image showing Y coordinate differences between left and right images"""
        try:
            print("DEBUG: Creating cocking debug image...")
            print(f"DEBUG: Left image shape: {left_cropped.shape}")
            print(f"DEBUG: Right image shape: {right_cropped.shape}")
            
            # Get dimensions of both images
            left_h, left_w = left_cropped.shape[:2]
            right_h, right_w = right_cropped.shape[:2]
            
            # Use the larger height to accommodate both images
            max_h = max(left_h, right_h)
            combined_img = np.zeros((max_h, left_w + right_w, 3), dtype=np.uint8)
            
            # Place left image
            combined_img[:left_h, :left_w] = left_cropped
            
            # Place right image (resize if necessary)
            if right_h != left_h:
                # Resize right image to match left image height
                right_resized = cv2.resize(right_cropped, (right_w, left_h))
                combined_img[:left_h, left_w:left_w + right_w] = right_resized
                print(f"DEBUG: Right image resized from {right_h}x{right_w} to {left_h}x{right_w}")
            else:
                combined_img[:right_h, left_w:left_w + right_w] = right_cropped
                print("DEBUG: Right image placed without resizing")
            
            # Draw Y coordinate lines and differences
            colors_map = {'C': (0, 255, 255), 'M': (255, 0, 255), 'Y': (255, 255, 0), 'S': (128, 128, 128)}
            
            print(f"DEBUG: Left results structure: {left_results}")
            print(f"DEBUG: Right results structure: {right_results}")
            
            for i, color in enumerate(['C', 'M', 'Y', 'S']):
                left_data = left_results.get(color)
                right_data = right_results.get(color)
                
                print(f"DEBUG: Processing color {color}")
                print(f"DEBUG: Left data for {color}: {left_data}")
                print(f"DEBUG: Right data for {color}: {right_data}")
                
                if left_data and right_data:
                    try:
                        left_y = int(left_data['bottom_left_px'][1])
                        right_y = int(right_data['bottom_left_px'][1])
                        print(f"DEBUG: Y coordinates - Left: {left_y}, Right: {right_y}")
                        
                        draw_color = colors_map.get(color, (255, 255, 255))
                        
                        # Draw horizontal lines at Y coordinates
                        cv2.line(combined_img, (0, left_y), (left_w-1, left_y), draw_color, 2)
                        cv2.line(combined_img, (left_w, right_y), (left_w + right_w-1, right_y), draw_color, 2)
                        
                        # Draw vertical connecting line
                        cv2.line(combined_img, (left_w-1, left_y), (left_w, right_y), draw_color, 1)
                        
                        # Calculate difference
                        y_diff_px = right_y - left_y
                        y_diff_mm = y_diff_px * mm_per_pixel_y
                        
                        # Add text labels
                        text_y = 30 + i * 25
                        cv2.putText(combined_img, f"{color}: {y_diff_mm:+.3f}mm", (10, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                        
                    except KeyError as e:
                        print(f"ERROR: Key error for {color}: {e}")
                        print(f"DEBUG: Left data keys: {list(left_data.keys()) if left_data else 'None'}")
                        print(f"DEBUG: Right data keys: {list(right_data.keys()) if right_data else 'None'}")
                        continue
            
            # Add titles
            cv2.putText(combined_img, "LEFT", (left_w//2-30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined_img, "RIGHT", (left_w + right_w//2-40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            print("DEBUG: Cocking debug image created successfully")
            return combined_img
            
        except Exception as e:
            print(f"ERROR: Failed to create cocking debug image: {str(e)}")
            return None
            
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
                color_map = {'C': (255, 255, 0), 'M': (255, 0, 255), 'Y': (0, 255, 255), 'S': (128, 128, 128)}
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
            color_map = {'C': (255, 255, 0), 'M': (255, 0, 255), 'Y': (0, 255, 255), 'S': (255, 255, 255)}
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
            color_map = {'C': (255, 255, 0), 'M': (255, 0, 255), 'Y': (0, 255, 255), 'S': (255, 255, 255)}
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

class CMYKAnalyzerGUI(QMainWindow):
    """Main GUI application"""
    
    def __init__(self):
        super().__init__()
        self.left_image_path = None
        self.right_image_path = None
        self.analysis_results = None
        self.worker = None
        self.camera_capture_count = 0  # Track camera captures for sequential assignment
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("CMYK Registration Analyzer")
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
        input_group = QGroupBox("📸 Image Input (Left & Right)")
        input_layout = QVBoxLayout()
        
        # Left/Right upload buttons
        upload_buttons_layout = QHBoxLayout()
        
        self.upload_left_btn = QPushButton("📁 Upload Left Image")
        self.upload_left_btn.clicked.connect(self.upload_left_image)
        self.upload_left_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; padding: 10px; background-color: #e8f5e8; }")
        upload_buttons_layout.addWidget(self.upload_left_btn)
        
        self.upload_right_btn = QPushButton("📁 Upload Right Image")
        self.upload_right_btn.clicked.connect(self.upload_right_image)
        self.upload_right_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; padding: 10px; background-color: #f5e8e8; }")
        upload_buttons_layout.addWidget(self.upload_right_btn)
        
        input_layout.addLayout(upload_buttons_layout)
        
        # Image status labels
        status_layout = QHBoxLayout()
        self.left_image_status = QLabel("❌ No Left Image")
        self.left_image_status.setStyleSheet("color: #666; font-size: 12px;")
        self.right_image_status = QLabel("❌ No Right Image")
        self.right_image_status.setStyleSheet("color: #666; font-size: 12px;")
        status_layout.addWidget(self.left_image_status)
        status_layout.addWidget(self.right_image_status)
        input_layout.addLayout(status_layout)
        
        # Camera capture widget (modified for sequential capture)
        self.camera_widget = CameraCaptureWidget()
        self.camera_widget.image_captured.connect(self.handle_camera_capture)
        input_layout.addWidget(self.camera_widget)
        
        # Image preview section
        preview_group = QGroupBox("📷 Image Preview")
        preview_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        preview_layout = QVBoxLayout()
        
        # Left image preview
        left_preview_layout = QHBoxLayout()
        left_preview_layout.addWidget(QLabel("Left:"))
        self.left_preview_label = QLabel("No image")
        self.left_preview_label.setMinimumSize(150, 100)
        self.left_preview_label.setMaximumSize(200, 150)
        self.left_preview_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f0f0f0; color: #666; font-size: 10px;")
        self.left_preview_label.setAlignment(Qt.AlignCenter)
        left_preview_layout.addWidget(self.left_preview_label)
        preview_layout.addLayout(left_preview_layout)
        
        # Right image preview
        right_preview_layout = QHBoxLayout()
        right_preview_layout.addWidget(QLabel("Right:"))
        self.right_preview_label = QLabel("No image")
        self.right_preview_label.setMinimumSize(150, 100)
        self.right_preview_label.setMaximumSize(200, 150)
        self.right_preview_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f0f0f0; color: #666; font-size: 10px;")
        self.right_preview_label.setAlignment(Qt.AlignCenter)
        right_preview_layout.addWidget(self.right_preview_label)
        preview_layout.addLayout(right_preview_layout)
        
        preview_group.setLayout(preview_layout)
        input_layout.addWidget(preview_group)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Analysis button
        self.analyze_btn = QPushButton("🔍 Analyze Left Image")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 12px; background-color: #e3f2fd; }")
        layout.addWidget(self.analyze_btn)
        
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
        

        
        # 4. Adjustment values tab (for practical use)
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
        self.adjustment_table.setRowCount(4)  # C, M, Y, S
        self.adjustment_table.setColumnCount(4)  # LATERAL, CIRCUM, COCKING, Status
        self.adjustment_table.setHorizontalHeaderLabels(['LATERAL (mm)', 'CIRCUM (mm)', 'COCKING (mm)', 'Status'])
        
        # Set row labels
        colors = ['C (Cyan)', 'M (Magenta)', 'Y (Yellow)', 'S (Special)']
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

• COCKING: Y coordinate difference (Right - Left)
  - Positive value: Right image is LOWER than Left
  - Negative value: Right image is HIGHER than Left

💡 TIP: Make small adjustments and re-analyze for best results.
        """
        
        instructions_detail = QLabel(instructions_text.strip())
        instructions_detail.setStyleSheet("font-size: 12px; color: #2c3e50; margin: 10px;")
        instructions_detail.setWordWrap(True)
        instructions_layout.addWidget(instructions_detail)
        instructions_group.setLayout(instructions_layout)
        
        self.adjustment_layout.addWidget(instructions_group)
        
        self.adjustment_tab.setLayout(self.adjustment_layout)
        self.result_tabs.addTab(self.adjustment_tab, "⚙️ Adjustment Values")
        
        # 4. Visualization tab
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
        
        self.show_cocking_btn = QPushButton("5. Cocking\n(Y Coordinate Diff)")
        self.show_cocking_btn.clicked.connect(lambda: self.show_debug_image('cocking'))
        self.show_cocking_btn.setStyleSheet("QPushButton { padding: 15px; font-size: 16px; font-weight: bold; }")
        
        # Arrange buttons in grid (3 columns, 2 rows)
        image_btn_layout.addWidget(self.show_preprocessed_btn, 0, 0)
        image_btn_layout.addWidget(self.show_cmyk_btn, 0, 1)
        image_btn_layout.addWidget(self.show_p_points_btn, 0, 2)
        image_btn_layout.addWidget(self.show_t_points_btn, 1, 0)
        image_btn_layout.addWidget(self.show_cocking_btn, 1, 1)
        
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
        
    def upload_left_image(self):
        """Upload left image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Left Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.load_left_image(file_path)
            
    def upload_right_image(self):
        """Upload right image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Right Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.load_right_image(file_path)
            
    def handle_camera_capture(self, image_path):
        """Handle camera capture - assign to left or right sequentially"""
        print(f"DEBUG: Camera captured image: {image_path}, count: {self.camera_capture_count}")
        
        if self.camera_capture_count % 2 == 0:
            # First capture -> Left
            self.load_left_image(image_path)
            print("DEBUG: Assigned to LEFT")
        else:
            # Second capture -> Right
            self.load_right_image(image_path)
            print("DEBUG: Assigned to RIGHT")
            
        self.camera_capture_count += 1
        
    def load_left_image(self, image_path):
        """Load left image and update status"""
        self.left_image_path = image_path
        self.left_image_status.setText(f"✅ Left: {os.path.basename(image_path)}")
        self.left_image_status.setStyleSheet("color: #2e7d32; font-size: 12px; font-weight: bold;")
        
        # Update image preview
        self.update_image_preview(image_path, self.left_preview_label, "Left")
        
        self.update_analysis_button_state()
        self.statusBar().showMessage(f"Left image loaded: {os.path.basename(image_path)}")
        print(f"DEBUG: Left image loaded: {image_path}")
        
    def load_right_image(self, image_path):
        """Load right image and update status"""
        self.right_image_path = image_path
        self.right_image_status.setText(f"✅ Right: {os.path.basename(image_path)}")
        self.right_image_status.setStyleSheet("color: #c62828; font-size: 12px; font-weight: bold;")
        
        # Update image preview
        self.update_image_preview(image_path, self.right_preview_label, "Right")
        
        self.statusBar().showMessage(f"Right image loaded: {os.path.basename(image_path)}")
        print(f"DEBUG: Right image loaded: {image_path}")
        
    def update_image_preview(self, image_path, preview_label, image_type):
        """Update image preview label with thumbnail"""
        try:
            # Load image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                preview_label.setText(f"❌ Failed to load\n{image_type} image")
                return
                
            # Resize image to fit preview label
            h, w = img.shape[:2]
            preview_w, preview_h = 150, 100
            
            # Calculate aspect ratio
            aspect = w / h
            if aspect > preview_w / preview_h:
                # Image is wider than preview
                new_w = preview_w
                new_h = int(preview_w / aspect)
            else:
                # Image is taller than preview
                new_h = preview_h
                new_w = int(preview_h * aspect)
            
            # Resize image
            resized = cv2.resize(img, (new_w, new_h))
            
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to QPixmap
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Set pixmap to preview label
            preview_label.setPixmap(pixmap)
            preview_label.setScaledContents(False)
            preview_label.setAlignment(Qt.AlignCenter)
            
            print(f"DEBUG: {image_type} image preview updated successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to update {image_type} image preview: {str(e)}")
            preview_label.setText(f"❌ Preview error\n{image_type}")
        
    def update_analysis_button_state(self):
        """Update analysis button state based on left image availability"""
        if self.left_image_path:
            self.analyze_btn.setEnabled(True)
            self.status_label.setText("Ready to analyze! Click 'Analyze Left Image' button.\n💡 Right image is optional for comparison.")
        else:
            self.analyze_btn.setEnabled(False)
            self.status_label.setText("Upload left image to enable analysis.\n💡 Right image is optional for comparison.")
        
    def start_analysis(self):
        """Start analysis on left image"""
        if not self.left_image_path:
            QMessageBox.warning(self, "Error", "Please load left image first.")
            return
            
        print(f"DEBUG: Starting analysis for LEFT image: {self.left_image_path}")  # Debug message
        
        # Stop any existing worker
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            
        # Update UI state
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Infinite progress
        self.status_label.setText("Analyzing left image...")
        self.analyze_btn.setEnabled(False)
        
        # Start worker thread with left and right images
        print(f"DEBUG: Creating AnalysisWorker with left: {self.left_image_path}")
        print(f"DEBUG: Creating AnalysisWorker with right: {self.right_image_path}")
        self.worker = AnalysisWorker(self.left_image_path, self.right_image_path)
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
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Analysis completed! Left image analyzed successfully.")
        
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
        self.analyze_btn.setEnabled(True)
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
        
        # 3. Adjustment values tab
        self.populate_adjustment_table(results['adjustments'])
        
        # 4. Visualization tab - show first image by default
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
        if hasattr(self, 'adjustment_table') and self.adjustment_table:
            self.adjustment_table.repaint()
        
        if hasattr(self, 'image_label') and self.image_label:
            self.image_label.repaint()
        
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.repaint()
        
        # Force overall refresh
        self.result_tabs.update()
        self.result_tabs.repaint()
        
        # Process all pending events
        QApplication.processEvents()
        
        print("DEBUG: UI refresh completed")

    def generate_overview_text(self, results):
        """Generate overview text"""
        text = "=== CMYK Registration Analysis Report ===\n\n"
        
        # Metadata
        meta = results['metadata']
        text += f"📅 Analysis Time: {meta['timestamp']}\n"
        text += f"📁 Left Image: {os.path.basename(meta.get('left_image_path', 'Unknown'))}\n"
        text += f"📁 Right Image: {os.path.basename(meta.get('right_image_path', 'None')) if meta.get('right_image_path') else 'None'}\n"

        text += f"🖼️ Image Size: {meta['image_size_px'][0]} x {meta['image_size_px'][1]} pixels\n"
        text += f"📐 mm per pixel: {meta['mm_per_pixel'][0]:.6f} x {meta['mm_per_pixel'][1]:.6f}\n"
        text += f"✅ Preprocessing: Square marker extraction completed\n\n"
        
        # Color-wise summary
        text += "=== Color-wise Analysis Summary ===\n\n"
        
        reg_results = results['registration']
        
        for color in ['C', 'M', 'Y', 'S']:
            color_desc = {'C': 'Cyan', 'M': 'Magenta', 'Y': 'Yellow', 'S': 'Special Color'}
            text += f"🎨 {color} ({color_desc[color]}):\n"
            
            # Registration results
            if reg_results.get(color):
                reg = reg_results[color]
                dx, dy = reg['movement_mm']
                text += f"  📍 Registration: ({dx:+.3f}, {dy:+.3f}) mm\n"
            else:
                text += f"  ❌ Registration: Detection failed\n"
                

                
            text += "\n"
            
        return text
        
    def generate_adjustment_text(self, adjustments):
        """Generate practical adjustment values text for field use"""
        text = "=== PRINTING MACHINE ADJUSTMENT VALUES ===\n\n"
        text += "🔧 Use these values to adjust your printing machine:\n\n"
        
        # Create a more readable table format
        text += "┌─────────┬──────────────┬─────────────┬────────┐\n"
        text += "│ Color   │ LATERAL (mm) │ CIRCUM (mm) │ Status │\n"
        text += "├─────────┼──────────────┼─────────────┼────────┤\n"
        
        for color in ['C', 'M', 'Y', 'S']:
            adj = adjustments.get(color, {})
            
            if adj.get('status') == 'OK':
                lateral = f"{adj['LATERAL']:+.3f}" if adj['LATERAL'] is not None else "N/A"
                circum = f"{adj['CIRCUM']:+.3f}" if adj['CIRCUM'] is not None else "N/A"
                status = "✅ OK"
            else:
                lateral = "N/A"
                circum = "N/A"
                status = "❌ Failed"
            
            text += f"│ {color:7} │ {lateral:12} │ {circum:11} │ {status:6} │\n"
        
        text += "└─────────┴──────────────┴─────────────┴────────┘\n"
        
        text += "\n" + "="*80 + "\n"
        text += "📋 ADJUSTMENT INSTRUCTIONS:\n\n"
        text += "• LATERAL: Adjust left-right position\n"
        text += "  - Positive value: Move RIGHT\n"
        text += "  - Negative value: Move LEFT\n\n"
        text += "• CIRCUM: Adjust up-down position\n"
        text += "  - Positive value: Move UP\n"
        text += "  - Negative value: Move DOWN\n\n"
        text += "💡 TIP: Make small adjustments and re-analyze for best results.\n"
        
        return text
        
    def populate_adjustment_table(self, adjustments):
        """Populate the adjustment table with results"""
        print(f"DEBUG: Populating adjustment table with: {adjustments}")  # Debug message
        colors = ['C', 'M', 'Y', 'S']
        
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
                f"📁 Left Image: {os.path.basename(meta.get('left_image_path', 'Unknown'))}",
                f"📁 Right Image: {os.path.basename(meta.get('right_image_path', 'None')) if meta.get('right_image_path') else 'None'}",
                f"🖼️ Image Size: {meta['image_size_px'][0]} x {meta['image_size_px'][1]} pixels",
                f"📐 Resolution: {meta['mm_per_pixel'][0]:.6f} x {meta['mm_per_pixel'][1]:.6f} mm/pixel",
                f"✅ Preprocessing: Square marker extraction completed"
            ]
        )
        self.overview_cards_layout.addWidget(meta_card)
        
        # Color summary cards
        reg_results = results['registration']
        adj_results = results['adjustments']
        
        # Create a grid layout for color cards
        colors_widget = QWidget()
        colors_layout = QGridLayout()
        
        # Extract special color name dynamically from results
        special_color_name = results.get('special_color_name', 'Special Color')
        
        color_names = {'C': 'Cyan', 'M': 'Magenta', 'Y': 'Yellow', 'S': special_color_name}
        color_emojis = {'C': '🔵', 'M': '🟣', 'Y': '🟡', 'S': '🟠'}
        
        for i, color in enumerate(['C', 'M', 'Y', 'S']):
            # Get data
            reg_data = reg_results.get(color)
            adj_data = adj_results.get(color)
            
            info_lines = []
            
            if reg_data:
                dx, dy = reg_data['movement_mm']
                info_lines.append(f"📍 Registration: ({dx:+.3f}, {dy:+.3f}) mm")
                info_lines.append(f"   → LATERAL: {dx:+.3f} mm ({'RIGHT' if dx > 0 else 'LEFT'})")
                info_lines.append(f"   → CIRCUM: {dy:+.3f} mm ({'UP' if dy > 0 else 'DOWN'})")
            else:
                info_lines.append("❌ Registration: Detection failed")
                
            if adj_data and adj_data.get('status') == 'OK':
                lateral = adj_data['LATERAL']
                circum = adj_data['CIRCUM']
                cocking = adj_data['COCKING']
                info_lines.append(f"🔧 Adjustments:")
                info_lines.append(f"   LATERAL: {lateral:+.3f} mm")
                info_lines.append(f"   CIRCUM: {circum:+.3f} mm")
                if cocking is not None:
                    info_lines.append(f"   COCKING: {cocking:+.3f} mm")
                else:
                    info_lines.append(f"   COCKING: N/A (no right image)")
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
        

        
    def show_debug_image(self, image_type):
        """Display debug image"""
        print(f"DEBUG: Showing debug image: {image_type}")
        
        if not self.analysis_results:
            self.image_label.setText("No analysis results available")
            self.image_label.update()
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
        elif image_type == 'cocking':
            debug_img = self.analysis_results.get('cocking_debug')
            if debug_img is None:
                # Check if right image was provided
                if self.right_image_path:
                    self.image_label.setText("Cocking analysis failed\n(Right image processing error)")
                else:
                    self.image_label.setText("Cocking analysis not available\n(No right image uploaded)")
                return
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
            'cocking': "5. Cocking - Y coordinate differences between Left and Right images"
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
                    if key in ['debug_reg', 'cocking_debug', 'preprocessed', 'cmyk_detection', 'p_points', 't_points']:
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