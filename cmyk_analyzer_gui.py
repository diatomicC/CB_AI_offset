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
        print(f"⚠️ Could not find enough C, M, Y boxes. Found: {list(known_boxes.keys())}")
        # 에러를 반환하지 않고 찾은 것들만이라도 계속 진행
        if len(known_boxes) == 0:
            print("❌ No CMY boxes found at all. Cannot continue.")
            return None, None
        print(f"🔄 Continuing with found boxes: {list(known_boxes.keys())}")
    
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
    # Avoid ranges that overlap with known CMY colors
    color_ranges = [
        # Red series
        ((0, 50, 50), (10, 255, 255)),
        ((170, 50, 50), (180, 255, 255)),
        # Orange series
        ((10, 50, 50), (25, 255, 255)),
        # Green series
        ((35, 50, 50), (85, 255, 255)),
        # Blue series - avoid overlap with Cyan
        ((100, 50, 50), (130, 255, 255)),
        # Purple series - avoid overlap with Magenta (140-170)
        ((130, 50, 50), (139, 255, 255)),  # Purple before Magenta
        ((171, 50, 50), (180, 255, 255)),  # Purple after Magenta
        # Pink series - more specific to avoid Magenta overlap
        ((140, 20, 50), (170, 40, 255)),   # Very light pink only
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
                    
                    # Check if too close to already known boxes or overlaps with known colors
                    too_close = False
                    
                    # First check distance to known boxes
                    for known_box in known_boxes.values():
                        known_center = known_box['center']
                        distance = np.sqrt((center[0] - known_center[0])**2 + 
                                        (center[1] - known_center[1])**2)
                        if distance < 50:  # Too close if within 50 pixels
                            too_close = True
                            print(f"❌ Special candidate too close to {known_box['color']} box (distance: {distance:.1f})")
                            break
                    
                    # Additional check: Test if this contour overlaps significantly with any known color
                    if not too_close:
                        contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(contour_mask, [contour], 255)
                        
                        for known_color, known_hsv_range in exclude_colors.items():
                            known_mask = cv2.inRange(hsv, np.array(known_hsv_range[0]), np.array(known_hsv_range[1]))
                            overlap = cv2.bitwise_and(contour_mask, known_mask)
                            overlap_area = cv2.countNonZero(overlap)
                            overlap_ratio = overlap_area / area
                            
                            if overlap_ratio > 0.3:  # More than 30% overlap
                                too_close = True
                                print(f"❌ Special candidate overlaps {overlap_ratio:.1%} with {known_color} color")
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

def classify_color_by_lab(img: np.ndarray, contour: np.ndarray) -> str:
    """
    Lab 색공간을 사용한 정확한 색상 분류
    
    Args:
        img: BGR 이미지
        contour: 색상 박스의 윤곽선
        
    Returns:
        str: 분류된 색상 ('C', 'M', 'Y', 'K')
    """
    # ROI 마스크 생성
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    # Lab 색공간으로 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    
    # ROI 내부 평균 색상 계산
    mean_L = cv2.mean(L, mask=mask)[0]
    mean_a = cv2.mean(a, mask=mask)[0]
    mean_b = cv2.mean(b, mask=mask)[0]
    
    print(f"🎨 Lab values: L={mean_L:.1f}, a={mean_a:.1f}, b={mean_b:.1f}")
    
    # Lab 값 기반 분류
    # L: 밝기 (0=검정, 100=흰색)
    # a: 빨강-초록 (음수=초록, 양수=빨강)  
    # b: 파랑-노랑 (음수=파랑, 양수=노랑)
    
    if mean_L < 50:  # 어두운 색상
        return 'S'  # Special color (검은색)
    elif mean_a > 20 and mean_b < 10:  # 빨강 성분 높고 노랑 성분 낮음
        return 'M'  # Magenta
    elif mean_b > 20 and mean_a < 10:  # 노랑 성분 높고 빨강 성분 낮음
        return 'Y'  # Yellow
    elif mean_a < -10 and mean_b < -10:  # 초록/파랑 성분
        return 'C'  # Cyan
    else:
        # 애매한 경우 RGB 기반 보조 판단
        mean_bgr = cv2.mean(img, mask=mask)[:3]
        b_val, g_val, r_val = mean_bgr
        
        if r_val > g_val and r_val > b_val:
            return 'M'  # 빨강 계열
        elif g_val > r_val and g_val > b_val:
            return 'C'  # 초록/청록 계열
        elif b_val > r_val and b_val > g_val:
            return 'C'  # 파랑 계열
        else:
            return 'Y'  # 노랑 계열

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
    print(f"🔍 DEBUG: registration_results keys: {list(registration_results.keys())}")
    adjustments = {}
    
    for color in ['C', 'M', 'Y', 'S']:
        # Get registration data
        reg_data = registration_results.get(color)
        cocking_data = cocking_results.get(color) if cocking_results else None
        
        print(f"🔍 DEBUG: {color} - reg_data: {reg_data}")
        
        if reg_data is None:
            print(f"❌ DEBUG: {color} - reg_data is None")
            adjustments[color] = {
                'LATERAL': None,
                'CIRCUM': None,
                'COCKING': None,
                'status': 'Detection failed'
            }
            continue
            
        # Extract movement values (in mm)
        if 'movement_mm' not in reg_data:
            print(f"❌ DEBUG: {color} - movement_mm not found in reg_data")
            adjustments[color] = {
                'LATERAL': None,
                'CIRCUM': None,
                'COCKING': None,
                'status': 'Missing movement data'
            }
            continue
            
        dx_mm, dy_mm = reg_data['movement_mm']
        print(f"✅ DEBUG: {color} - movement_mm: ({dx_mm}, {dy_mm})")
        
        # Convert to practical adjustment values
        # LATERAL = X-axis movement (left-right)
        # Right = +, Left = -
        lateral_mm = dx_mm
        
        # CIRCUM = Y-axis movement (up-down)
        # Up = +, Down = -
        circum_mm = dy_mm
        
        # COCKING = Y coordinate difference (right - left)
        cocking_mm = None
        if cocking_data and isinstance(cocking_data, dict) and 'cocking_mm' in cocking_data:
            cocking_mm = cocking_data['cocking_mm']
        
        # Determine status based on available data
        status = 'OK'
        if cocking_mm is None:
            status = 'Cocking data missing'
        elif abs(cocking_mm) > 0.5:  # 0.5mm 이상이면 주의
            status = 'Warning'
        
        adjustments[color] = {
            'LATERAL': round(lateral_mm, 3),
            'CIRCUM': round(circum_mm, 3),
            'COCKING': round(cocking_mm, 3) if cocking_mm is not None else None,
            'status': status
        }
        
        print(f"✅ DEBUG: {color} - Final adjustments: {adjustments[color]}")
    
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
            # Based on actual image analysis, these are the correct HSV ranges
            base_colors = {
                'C': ((90,50,50),(140,255,255)),   # Blue (Cyan in printing) - 더 넓은 범위
                'M': ((150,80,80),(170,255,255)),   # Magenta - 더 높은 채도로 핑크색만 감지
                'Y': ((15,60,60),(35,255,255)),     # Orange (Yellow in printing)
            }
            
            # Detect special color
            self.progress.emit("Detecting special color...")
            special_hsv_range, special_color_name = detect_special_color(cropped, base_colors)
            
            # Complete color ranges (including special color if detected)
            HSV = base_colors.copy()
            if special_hsv_range is not None:
                HSV['S'] = special_hsv_range  # Mark special color as 'S'
                self.special_color_name = special_color_name
                print(f"✅ Special color detected: {special_color_name}")
            else:
                print("⚠️  Special color not detected. Continuing with CMY colors only.")
                self.special_color_name = "None"
            
            # Target coordinates (bottom-left origin)
            # Each color's T point should be at the corresponding corner of its own color box
            # Based on 2x2 layout: Top-left(S), Top-right(C), Bottom-left(M), Bottom-right(Y)
            target_coords = {}
            if 'S' in HSV:
                target_coords['S'] = (w_px*4/10, h_px*4/10)           # Special(Top-left box)
            target_coords['C'] = (w_px*6/10, h_px*4/10)           # Blue(Top-right box) 
            target_coords['M'] = (w_px*4/10, h_px*6/10)           # Magenta(Bottom-left box) 
            target_coords['Y'] = (w_px*6/10, h_px*6/10)            # Yellow(Bottom-right box)
            
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
            
            # Helper functions for robust P point selection
            def _is_inside(x: float, y: float, w: int, h: int, margin: int = 0) -> bool:
                return margin <= x < (w - margin) and margin <= y < (h - margin)

            def _get_corner_index_for_color(color_name: str) -> int:
                # rect order: [top-left(0), top-right(1), bottom-right(2), bottom-left(3)]
                # P point corner mapping for each color:
                mapping = {
                    'S': 2,  # 우측하단 (bottom-right)
                    'C': 3,  # 좌측하단 (bottom-left)
                    'M': 1,  # 우측상단 (top-right)
                    'Y': 0,  # 좌측상단 (top-left)
                }
                return mapping.get(color_name, 3)

            def _safe_point_towards_center(corners: np.ndarray, corner_idx: int, fraction: float = 0.2) -> tuple[float, float]:
                # Move from the chosen corner towards the box center to guarantee it's inside
                cx = float(np.mean(corners[:, 0]))
                cy = float(np.mean(corners[:, 1]))
                x0 = float(corners[corner_idx][0])
                y0 = float(corners[corner_idx][1])
                x = x0 + (cx - x0) * fraction
                y = y0 + (cy - y0) * fraction
                return x, y

            for color, hsv_range in HSV.items():
                print(f"🔍 DEBUG: Processing color {color} with HSV range {hsv_range}")
                # Use corner detection to get all 4 corners
                rect = detect_square_corners(cropped, hsv_range, min_area_ratio=0.7)
                px_px = py_px = None
                
                if rect is not None:
                    # rect: [top-left(0), top-right(1), bottom-right(2), bottom-left(3)]
                    # Select the specific corner for each color
                    corner_idx = _get_corner_index_for_color(color)
                    candidate_x, candidate_y = rect[corner_idx]
                    
                    # Check if the corner is inside the image bounds
                    if _is_inside(candidate_x, candidate_y, cropped.shape[1], cropped.shape[0], margin=5):
                        # Use the corner directly
                        px_px, py_px = float(candidate_x), float(candidate_y)
                        print(f"✅ {color}: Using {['top-left', 'top-right', 'bottom-right', 'bottom-left'][corner_idx]} corner at ({px_px:.1f}, {py_px:.1f})")
                    else:
                        # Move slightly inward from the corner
                        candidate_x, candidate_y = _safe_point_towards_center(rect, corner_idx, fraction=0.15)
                        px_px, py_px = float(candidate_x), float(candidate_y)
                        print(f"⚠️ {color}: Corner outside bounds, using safe inward point at ({px_px:.1f}, {py_px:.1f})")
                else:
                    # Fallback to legacy bottom-left detector if corner detection fails
                    print(f"⚠️ {color}: Corner detection failed, using fallback method")
                    bl = detect_bottom_left(cropped, hsv_range, min_area_ratio=0.7)
                    if bl is not None:
                        px_px, py_px = float(bl[0]), float(bl[1])
                        print(f"✅ {color}: Fallback detection at ({px_px:.1f}, {py_px:.1f})")

                if px_px is None or py_px is None:
                    print(f"❌ {color}: P point detection failed completely")
                    results_reg[color] = None
                    continue

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
                
                print(f"🔍 DEBUG: {color} - P({px_px:.1f}, {py_px:.1f}) -> T({tx_px:.1f}, {ty_px:.1f})")
                print(f"🔍 DEBUG: {color} - movement_mm: ({dx_mm:.3f}, {dy_mm:.3f})")
                
                results_reg[color] = {
                    'P_coord_mm': (round(px_mm, 3), round(py_mm, 3)),
                    'T_coord_mm': (round(tx_mm, 3), round(ty_mm, 3)),
                    'movement_mm': (round(dx_mm, 3), round(dy_mm, 3)),
                    'P_coord_px': (px_px, py_px)  # Add pixel coordinates for cocking
                }
                
                print(f"✅ DEBUG: {color} - results_reg[{color}] = {results_reg[color]}")
                
                # Debug visualization for registration image
                px_int, py_int = int(round(px_px)), int(round(py_px))
                cv2.circle(debug_reg, (px_int, py_int), 8, (0,255,0), -1)
                cv2.circle(debug_reg, (px_int, py_int), 10, (0,0,0), 2)
                cv2.putText(debug_reg, f"P{color}({px_mm:.2f},{py_mm:.2f})", 
                           (px_int+15, py_int-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(debug_reg, f"Move({dx_mm:.2f},{dy_mm:.2f})", 
                           (px_int+15, py_int+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            self.progress.emit("Analysis completed!")
            
            # Analyze right image for cocking calculation if available
            cocking_results = {}
            cocking_debug_img = None
            right_results_reg = {}
            right_cropped = None
            
            if self.right_image_path:
                self.progress.emit("Analyzing right image for cocking calculation...")
                
                # Process right image the same way as left image
                right_orig = cv2.imread(self.right_image_path)
                if right_orig is not None:
                    # Extract square marker from right image
                    right_preprocessed = extract_robust_square_marker(right_orig)
                    if right_preprocessed is not None:
                        # Extract CMYK marker region from right image
                        right_cropped = extract_marker(right_preprocessed)
                        if right_cropped is not None:
                            # Apply same P and T calculation logic to right image
                            right_h_px, right_w_px = right_cropped.shape[:2]
                            right_mm_per_pixel_x = 5.0 / right_w_px
                            right_mm_per_pixel_y = 5.0 / right_h_px
                            
                            # Use same target coordinates for right image
                            right_target_coords = {
                                'S': (right_w_px*4/10, right_h_px*4/10),           # Special(Top-left box)
                                'C': (right_w_px*6/10, right_h_px*4/10),           # Blue(Top-right box) 
                                'M': (right_w_px*4/10, right_h_px*6/10),           # Magenta(Bottom-left box) 
                                'Y': (right_w_px*6/10, right_h_px*6/10)            # Yellow(Bottom-right box) 
                            }
                            
                            # Helper functions (same as left image)
                            def _is_inside_right(x: float, y: float, w: int, h: int, margin: int = 0) -> bool:
                                return margin <= x < (w - margin) and margin <= y < (h - margin)

                            def _get_corner_index_for_color_right(color_name: str) -> int:
                                # Same corner mapping as left image
                                mapping = {
                                    'S': 2,  # 우측하단 (bottom-right)
                                    'C': 3,  # 좌측하단 (bottom-left)
                                    'M': 1,  # 우측상단 (top-right)
                                    'Y': 0,  # 좌측상단 (top-left)
                                }
                                return mapping.get(color_name, 3)

                            def _safe_point_towards_center_right(corners: np.ndarray, corner_idx: int, fraction: float = 0.2) -> tuple[float, float]:
                                cx = float(np.mean(corners[:, 0]))
                                cy = float(np.mean(corners[:, 1]))
                                x0 = float(corners[corner_idx][0])
                                y0 = float(corners[corner_idx][1])
                                x = x0 + (cx - x0) * fraction
                                y = y0 + (cy - y0) * fraction
                                return x, y
                            
                            for color, hsv_range in HSV.items():
                                # Use corner detection to get all 4 corners (same as left)
                                rect = detect_square_corners(right_cropped, hsv_range, min_area_ratio=0.7)
                                px_px = py_px = None
                                
                                if rect is not None:
                                    # Select the specific corner for each color
                                    corner_idx = _get_corner_index_for_color_right(color)
                                    candidate_x, candidate_y = rect[corner_idx]
                                    
                                    # Check if the corner is inside the image bounds
                                    if _is_inside_right(candidate_x, candidate_y, right_cropped.shape[1], right_cropped.shape[0], margin=5):
                                        # Use the corner directly
                                        px_px, py_px = float(candidate_x), float(candidate_y)
                                        print(f"✅ RIGHT {color}: Using {['top-left', 'top-right', 'bottom-right', 'bottom-left'][corner_idx]} corner at ({px_px:.1f}, {py_px:.1f})")
                                    else:
                                        # Move slightly inward from the corner
                                        candidate_x, candidate_y = _safe_point_towards_center_right(rect, corner_idx, fraction=0.15)
                                        px_px, py_px = float(candidate_x), float(candidate_y)
                                        print(f"⚠️ RIGHT {color}: Corner outside bounds, using safe inward point at ({px_px:.1f}, {py_px:.1f})")
                                else:
                                    # Fallback to legacy bottom-left detector
                                    print(f"⚠️ RIGHT {color}: Corner detection failed, using fallback method")
                                    bl = detect_bottom_left(right_cropped, hsv_range, min_area_ratio=0.7)
                                    if bl is not None:
                                        px_px, py_px = float(bl[0]), float(bl[1])
                                        print(f"✅ RIGHT {color}: Fallback detection at ({px_px:.1f}, {py_px:.1f})")

                                if px_px is None or py_px is None:
                                    print(f"❌ RIGHT {color}: P point detection failed completely")
                                    right_results_reg[color] = None
                                    continue

                                px_bl, py_bl = pixel_to_bottom_left_coord(px_px, py_px, right_h_px)
                                
                                tx_px, ty_px = right_target_coords[color]
                                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, right_h_px)
                                
                                right_results_reg[color] = {
                                    'P_coord_mm': (px_bl * right_mm_per_pixel_x, py_bl * right_mm_per_pixel_y),
                                    'T_coord_mm': (tx_bl * right_mm_per_pixel_x, ty_bl * right_mm_per_pixel_y),
                                    'bottom_left_px': [px_px, py_px]
                                }
                            
                            # Now calculate cocking using both left and right results
                            cocking_results, cocking_debug_img = self.calculate_cocking_from_results(
                                results_reg, right_results_reg, cropped, right_cropped, mm_per_pixel_y
                            )
                        else:
                            print("❌ Could not extract CMYK marker from right image")
                    else:
                        print("❌ Could not find square marker in right image")
                else:
                    print("❌ Could not read right image file")
            
            # Calculate adjustment values (now includes cocking)
            adjustments = calculate_adjustment_values(results_reg, cocking_results)
            
            # Create additional visualization images (side-by-side if right available)
            preprocessed_img = self.create_preprocessed_image(preprocessed, right_preprocessed)
            cmyk_detection_img = self.create_cmyk_detection_image(cropped, right_cropped, HSV, results_reg, right_results_reg)
            p_points_img = self.create_p_points_image(cropped, right_cropped, results_reg, right_results_reg, h_px)
            t_points_img = self.create_t_points_image(cropped, right_cropped, target_coords, h_px)
            
            # Return results
            results = {
                'registration': results_reg,
                'cocking': cocking_results,  # Cocking values (Y coordinate differences)
                'adjustments': adjustments,  # LATERAL, CIRCUM, COCKING values
                'debug_reg': debug_reg,
                'cocking_debug': cocking_debug_img,  # Cocking visualization
                'preprocessed': preprocessed_img,  # Preprocessed image (side-by-side)
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
            print(f"⚠️  Analysis error: {str(e)}")
            # 에러가 발생해도 부분적인 결과라도 보여주기
            if 'results' in locals() and results:
                print("🔄 Returning partial results despite error...")
                self.finished.emit(results)
            else:
                self.error.emit(f"Critical error during analysis: {str(e)}")
            
    def calculate_cocking_from_results(self, left_results_reg, right_results_reg, left_cropped, right_cropped, mm_per_pixel_y):
        """
        Calculate cocking values by comparing Y coordinates between left and right images.
        
        Cocking Algorithm:
        1. Use already processed left image P points from left_results_reg
        2. Use already processed right image P points from right_results_reg
        3. Compare Y positions of corresponding P points between left and right
        4. Calculate difference: right_y - left_y (positive = right is lower)
        
        Args:
            left_results_reg: Results from left image analysis (already processed)
            right_results_reg: Results from right image analysis (already processed)
            left_cropped: Processed left image for visualization
            right_cropped: Processed right image for visualization
            mm_per_pixel_y: Y-axis conversion factor
            
        Returns:
            tuple: (cocking_results, debug_image)
        """
        try:
            print("=" * 50)
            print("COCKING ALGORITHM - USING PRE-PROCESSED RESULTS")
            print("=" * 50)
            
            # Step 1: Validate input data
            print("STEP 1: Validating input data...")
            if not left_results_reg or not right_results_reg:
                print("❌ ERROR: Missing left or right image results")
                return {}, None
            print(f"✅ Left results: {list(left_results_reg.keys())}")
            print(f"✅ Right results: {list(right_results_reg.keys())}")
            
            # Step 2: Prepare left image results for comparison
            print("STEP 2: Preparing left image P points for comparison...")
            left_results_cocking = {}
            
            for color, data in left_results_reg.items():
                if data and 'P_coord_px' in data:
                    # Use the pixel coordinates directly from left image analysis
                    px_px, py_px = data['P_coord_px']
                    
                    left_results_cocking[color] = {
                        'bottom_left_px': [px_px, py_px],
                        'color': color
                    }
                    print(f"    ✅ {color}: P point at ({px_px:.1f}, {py_px:.1f}) pixels (from left analysis)")
                else:
                    print(f"    ❌ {color}: No valid data from left image")
                    left_results_cocking[color] = None
            
            # Step 3: Calculate cocking for each color
            print("STEP 3: Calculating cocking differences...")
            cocking_results = {}
            
            print("\nCOCKING ANALYSIS RESULTS:")
            print("-" * 40)
            
            for color in ['C', 'M', 'Y', 'S']:
                left_data = left_results_cocking.get(color)
                right_data = right_results_reg.get(color)
                
                if left_data and right_data and left_data is not None and right_data is not None:
                    try:
                        # Extract Y coordinates (bottom-left coordinate system)
                        left_y_px = left_data['bottom_left_px'][1]
                        right_y_px = right_data['bottom_left_px'][1]
                        
                        # Calculate Y difference in pixels
                        # Positive = right is lower than left
                        # Negative = right is higher than left
                        y_diff_px = right_y_px - left_y_px
                        
                        # Convert to millimeters
                        y_diff_mm = y_diff_px * mm_per_pixel_y
                        
                        cocking_results[color] = {
                            'left_y_px': round(left_y_px, 2),
                            'right_y_px': round(right_y_px, 2),
                            'y_diff_px': round(y_diff_px, 2),
                            'cocking_mm': round(y_diff_mm, 3),
                            'status': 'OK'
                        }
                        
                        status_symbol = "🔺" if y_diff_mm > 0 else "🔻" if y_diff_mm < 0 else "➡️"
                        print(f"{color}: Left={left_y_px:.1f}px, Right={right_y_px:.1f}px → {status_symbol} {y_diff_mm:+.3f}mm")
                        
                    except Exception as e:
                        print(f"❌ {color}: Calculation failed - {str(e)}")
                        cocking_results[color] = {
                            'status': 'Error',
                            'error': str(e)
                        }
                else:
                    missing = []
                    if not left_data or left_data is None:
                        missing.append("left")
                    if not right_data or right_data is None:
                        missing.append("right")
                    
                    print(f"❌ {color}: Missing data from {', '.join(missing)} image(s)")
                    cocking_results[color] = {
                        'status': 'Missing data',
                        'missing': missing
                    }
            
            # Step 4: Create debug visualization
            print("STEP 4: Creating cocking debug visualization...")
            debug_img = self.create_cocking_debug_image(
                left_cropped, right_cropped, 
                left_results_cocking, right_results_reg, 
                mm_per_pixel_y
            )
            
            print("=" * 50)
            print("COCKING ANALYSIS COMPLETED")
            print("=" * 50)
            
            return cocking_results, debug_img
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in cocking calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}, None
    
    def create_cocking_debug_image(self, left_cropped, right_cropped, left_results, right_results, mm_per_pixel_y):
        """
        Create comprehensive debug visualization for cocking analysis.
        Shows side-by-side comparison of left and right images with P points and Y differences.
        """
        try:
            print("Creating enhanced cocking debug visualization...")
            
            # Get dimensions
            left_h, left_w = left_cropped.shape[:2]
            right_h, right_w = right_cropped.shape[:2]
            
            # Create side-by-side layout with extra space for annotations
            margin = 100  # Space for labels and measurements
            max_h = max(left_h, right_h) + margin
            combined_w = left_w + right_w + 40  # 40px gap between images
            
            # Initialize combined image
            combined_img = np.zeros((max_h, combined_w, 3), dtype=np.uint8)
            combined_img.fill(50)  # Dark gray background
            
            # Place left image
            y_offset = margin // 2
            combined_img[y_offset:y_offset + left_h, :left_w] = left_cropped
            
            # Place right image with gap
            right_x_offset = left_w + 40
            combined_img[y_offset:y_offset + right_h, right_x_offset:right_x_offset + right_w] = right_cropped
            
            # Add image labels
            cv2.putText(combined_img, "LEFT IMAGE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined_img, "RIGHT IMAGE", (right_x_offset + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw P points and connections
            # Based on actual image colors for better visualization
            colors_rgb = {
                'C': (255, 0, 0),      # Blue (Cyan) -> Red for visibility
                'M': (255, 0, 255),    # Magenta
                'Y': (0, 165, 255),    # Orange (Yellow) -> Orange for visibility  
                'S': (0, 255, 255)     # Yellow (Special) -> Yellow for visibility
            }
            
            measurement_y = max_h - 60  # Y position for measurements text
            
            for i, color in enumerate(['C', 'M', 'Y', 'S']):
                left_data = left_results.get(color)
                right_data = right_results.get(color)
                color_rgb = colors_rgb.get(color, (255, 255, 255))
                
                if left_data and right_data and left_data is not None and right_data is not None:
                    # Get coordinates
                    left_x, left_y = left_data['bottom_left_px']
                    right_x, right_y = right_data['bottom_left_px']
                    
                    # Adjust coordinates for display
                    left_display_x = int(left_x)
                    left_display_y = int(left_y + y_offset)
                    right_display_x = int(right_x + right_x_offset)
                    right_display_y = int(right_y + y_offset)
                    
                    # Draw P points
                    cv2.circle(combined_img, (left_display_x, left_display_y), 6, color_rgb, -1)
                    cv2.circle(combined_img, (left_display_x, left_display_y), 8, (255, 255, 255), 2)
                    cv2.circle(combined_img, (right_display_x, right_display_y), 6, color_rgb, -1)
                    cv2.circle(combined_img, (right_display_x, right_display_y), 8, (255, 255, 255), 2)
                    
                    # Label P points
                    cv2.putText(combined_img, f"P{color}", 
                               (left_display_x + 12, left_display_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
                    cv2.putText(combined_img, f"P{color}", 
                               (right_display_x + 12, right_display_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
                    
                    # Draw horizontal line connecting P points (for Y comparison)
                    line_y = (left_display_y + right_display_y) // 2
                    cv2.line(combined_img, (left_display_x, line_y), 
                            (right_display_x, line_y), color_rgb, 2)
                    
                    # Calculate and display Y difference
                    y_diff_px = right_y - left_y
                    y_diff_mm = y_diff_px * mm_per_pixel_y
                    
                    # Add measurement text
                    measurement_text = f"{color}: {y_diff_mm:+.3f}mm"
                    text_x = 10 + (i * 120)
                    cv2.putText(combined_img, measurement_text, 
                               (text_x, measurement_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)
                    
                    # Add direction arrow in the middle of connection line
                    mid_x = (left_display_x + right_display_x) // 2
                    mid_y = line_y
                    
                    if abs(y_diff_mm) > 0.001:  # Only show arrow if difference is significant
                        arrow_color = (0, 255, 0) if y_diff_mm > 0 else (0, 0, 255)  # Green up, Red down
                        if y_diff_mm > 0:
                            # Right is lower - draw down arrow
                            cv2.arrowedLine(combined_img, (mid_x, mid_y - 10), (mid_x, mid_y + 10), arrow_color, 2)
                        else:
                            # Right is higher - draw up arrow
                            cv2.arrowedLine(combined_img, (mid_x, mid_y + 10), (mid_x, mid_y - 10), arrow_color, 2)
                else:
                    # Show missing data
                    measurement_text = f"{color}: N/A"
                    text_x = 10 + (i * 120)
                    cv2.putText(combined_img, measurement_text, 
                               (text_x, measurement_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Add legend
            legend_y = max_h - 30
            cv2.putText(combined_img, "Green Arrow: Right Lower | Red Arrow: Right Higher", 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print("✅ Cocking debug image created successfully")
            return combined_img
            
        except Exception as e:
            print(f"❌ Error creating cocking debug image: {str(e)}")
            # Return simple side-by-side image as fallback
            try:
                left_h, left_w = left_cropped.shape[:2]
                right_h, right_w = right_cropped.shape[:2]
                max_h = max(left_h, right_h)
                combined_img = np.zeros((max_h, left_w + right_w, 3), dtype=np.uint8)
                combined_img[:left_h, :left_w] = left_cropped
                combined_img[:right_h, left_w:] = right_cropped
                return combined_img
            except:
                return None

            
    def create_preprocessed_image(self, left_preprocessed, right_preprocessed):
        """Create side-by-side preprocessed image showing square marker extraction"""
        if left_preprocessed is None:
            return None
            
        left_h, left_w = left_preprocessed.shape[:2]
        has_right = right_preprocessed is not None
        
        if has_right:
            right_h, right_w = right_preprocessed.shape[:2]
            if (right_h, right_w) != (left_h, left_w):
                right_preprocessed = cv2.resize(right_preprocessed, (left_w, left_h))
        
        # Canvas
        gap = 40
        label_margin = 30
        canvas_w = left_w if not has_right else left_w * 2 + gap
        canvas_h = left_h + label_margin
        result_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        result_img.fill(50)
        
        # Place images
        result_img[label_margin:label_margin+left_h, :left_w] = left_preprocessed
        cv2.putText(result_img, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        if has_right:
            x_off = left_w + gap
            result_img[label_margin:label_margin+left_h, x_off:x_off+left_w] = right_preprocessed
            cv2.putText(result_img, "RIGHT", (x_off+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        return result_img
    
    def order_points_for_detection(self, pts):
        """Sort four points in clockwise order (same as color_registration_analysis.py)"""
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect = np.zeros((4,2), dtype="float32")
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right  
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        return rect
            
    def create_cmyk_detection_image(self, left_cropped, right_cropped, HSV, left_results=None, right_results=None):
        """Create image showing CMYK color detection results for left and right (if available)"""
        left_h, left_w = left_cropped.shape[:2]
        has_right = right_cropped is not None
        if has_right:
            right_h, right_w = right_cropped.shape[:2]
            if (right_h, right_w) != (left_h, left_w):
                right_cropped = cv2.resize(right_cropped, (left_w, left_h))
        
        # Canvas
        gap = 40
        label_margin = 30
        canvas_w = left_w if not has_right else left_w * 2 + gap
        canvas_h = left_h + label_margin
        result_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        result_img.fill(50)
        
        # Place images
        result_img[label_margin:label_margin+left_h, :left_w] = left_cropped
        cv2.putText(result_img, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if has_right:
            x_off = left_w + gap
            result_img[label_margin:label_margin+left_h, x_off:x_off+left_w] = right_cropped
            cv2.putText(result_img, "RIGHT", (x_off+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        color_map = {'C': (255, 0, 0), 'M': (255, 0, 255), 'Y': (0, 165, 255), 'S': (0, 255, 255)}
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        
        def draw_for(img, x_offset=0, results=None):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            print(f"🔍 CMYK Detection called with results: {results is not None}")
            if results:
                print(f"🔍 Results keys: {list(results.keys())}")
            
            # First draw contours using HSV ranges
            detected_positions = {}
            for color, hsv_range in HSV.items():
                mask = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                areas = [cv2.contourArea(c) for c in cnts]
                max_area = max(areas)
                min_area_threshold = max_area * 0.3
                large_cnts = [c for c, area in zip(cnts, areas) if area >= min_area_threshold]
                draw_color = color_map.get(color, (255,255,255))
                # offset contours for canvas
                for c in large_cnts:
                    cv2.drawContours(result_img, [c + np.array([x_offset, label_margin])], -1, draw_color, 3)
                
                # Store position and classify using Lab color space
                if large_cnts:
                    largest_cnt = max(large_cnts, key=cv2.contourArea)
                    
                    # Lab 색공간 기반 정확한 색상 분류
                    actual_color = classify_color_by_lab(img, largest_cnt)
                    print(f"🎨 {color} HSV detection -> Lab classification: {actual_color}")
                    
                    # Use same corner detection as P points
                    hull = cv2.convexHull(largest_cnt)
                    eps = 0.02 * cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, eps, True).reshape(-1,2)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        # Order points same as detect_square_corners
                        rect = self.order_points_for_detection(approx)
                        # Get specific corner for each color (same as P points logic)
                        corner_mapping = {
                            'S': 2,  # bottom-right (Special color)
                            'C': 3,  # bottom-left  
                            'M': 1,  # top-right
                            'Y': 0,  # top-left
                        }
                        corner_idx = corner_mapping.get(actual_color, 3)
                        cx = int(rect[corner_idx][0]) + x_offset
                        cy = int(rect[corner_idx][1]) + label_margin
                        detected_positions[actual_color] = (cx, cy)  # Use actual color as key
                        print(f"🔍 {actual_color}: Using corner {corner_idx} at ({cx-x_offset},{cy-label_margin})")
                    else:
                        # Fallback to center
                        M = cv2.moments(largest_cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"]/M["m00"]) + x_offset
                            cy = int(M["m01"]/M["m00"]) + label_margin
                            detected_positions[actual_color] = (cx, cy)  # Use actual color as key
                            print(f"🔍 {actual_color}: Using center fallback at ({cx-x_offset},{cy-label_margin})")
            
            # Now label based on actual detection results if available
            if results:
                # Ignore HSV detection completely, use only actual analysis results
                print(f"🔍 CMYK Detection Debug - Image offset: {x_offset}")
                print(f"🔍 Available results: {list(results.keys())}")
                
                # Get image dimensions
                img_h, img_w = img.shape[:2]
                mm_per_pixel_x = 5.0 / img_w
                mm_per_pixel_y = 5.0 / img_h
                
                # Draw labels directly at actual P point positions (ignore HSV detection)
                for actual_color, reg_data in results.items():
                    if not reg_data:
                        continue
                    
                    # Get actual detected position from analysis results
                    # Try different data structure keys
                    px_px = py_px = None
                    if 'P_coord_px' in reg_data:
                        px_px, py_px = reg_data['P_coord_px']
                    elif 'bottom_left_px' in reg_data:
                        px_px, py_px = reg_data['bottom_left_px']
                    else:
                        print(f"❌ {actual_color}: No pixel coordinates found in {list(reg_data.keys())}")
                        continue
                    
                    # Convert to canvas coordinates
                    px_cv = int(px_px) + x_offset
                    py_cv = int(py_px) + label_margin
                    
                    print(f"🔍 {actual_color}: coords=({px_px:.1f},{py_px:.1f}) -> canvas=({px_cv},{py_cv})")
                    
                    # Draw label directly at actual position
                    draw_color = color_map.get(actual_color, (255,255,255))
                    cv2.putText(result_img, f"{actual_color}", (px_cv-10, py_cv), cv2.FONT_HERSHEY_SIMPLEX, 1.0, draw_color, 3)
                    print(f"    ✅ Drawing {actual_color} label at ({px_cv},{py_cv})")
                        
                print(f"🔍 Used actual P point positions for labeling")
                
                # Clear detected_positions to avoid double labeling
                detected_positions.clear()
            else:
                # Fallback to Lab-based classification labeling
                print(f"🔍 Using Lab-based classification for labeling")
                for color, (cx, cy) in detected_positions.items():
                    # Use the Lab-classified color directly
                    draw_color = color_map.get(color, (255,255,255))
                    cv2.putText(result_img, f"{color}", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, draw_color, 3)
                    print(f"    ✅ Drawing Lab-classified {color} label at ({cx},{cy})")
        
        draw_for(left_cropped, 0, left_results)
        if has_right:
            draw_for(right_cropped, left_w + gap, right_results)
        
        return result_img
    
    def create_p_points_image(self, left_cropped, right_cropped, left_results_reg, right_results_reg, h_px):
        """Create image showing P points for left and right (if available)"""
        left_h, left_w = left_cropped.shape[:2]
        has_right = right_cropped is not None and bool(right_results_reg)
        if has_right:
            right_h, right_w = right_cropped.shape[:2]
            if (right_h, right_w) != (left_h, left_w):
                right_cropped = cv2.resize(right_cropped, (left_w, left_h))
        gap = 40
        label_margin = 30
        canvas_w = left_w if not has_right else left_w*2 + gap
        canvas_h = left_h + label_margin
        result_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        result_img.fill(50)
        result_img[label_margin:label_margin+left_h, :left_w] = left_cropped
        cv2.putText(result_img, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if has_right:
            x_off = left_w + gap
            result_img[label_margin:label_margin+left_h, x_off:x_off+left_w] = right_cropped
            cv2.putText(result_img, "RIGHT", (x_off+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        def draw_points(results, mm_w, mm_h, x_off=0):
            for color, reg_data in results.items():
                if not reg_data:
                    continue
                px_mm, py_mm = reg_data['P_coord_mm']
                px_bl = px_mm / mm_w
                py_bl = py_mm / mm_h
                px_cv = int(px_bl) + x_off
                py_cv = int(h_px - py_bl)
                color_map = {'C': (255, 255, 0), 'M': (255, 0, 255), 'Y': (0, 255, 255), 'S': (255, 255, 255)}
                draw_color = color_map.get(color, (255,255,255))
                cv2.circle(result_img, (px_cv, py_cv+label_margin), 12, draw_color, -1)
                cv2.circle(result_img, (px_cv, py_cv+label_margin), 15, (0,0,0), 3)
                cv2.putText(result_img, f"P{color}", (px_cv+20, py_cv+label_margin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)
                cv2.putText(result_img, f"({px_mm:.2f},{py_mm:.2f})", (px_cv+20, py_cv+label_margin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        mm_per_pixel_x_left = 5.0 / left_cropped.shape[1]
        mm_per_pixel_y_left = 5.0 / left_cropped.shape[0]
        draw_points(left_results_reg, mm_per_pixel_x_left, mm_per_pixel_y_left, 0)
        if has_right:
            mm_per_pixel_x_right = 5.0 / right_cropped.shape[1]
            mm_per_pixel_y_right = 5.0 / right_cropped.shape[0]
            draw_points(right_results_reg, mm_per_pixel_x_right, mm_per_pixel_y_right, left_w + gap)
        
        return result_img
    
    def create_t_points_image(self, left_cropped, right_cropped, target_coords, h_px):
        """Create image showing T points side-by-side if right available"""
        left_h, left_w = left_cropped.shape[:2]
        has_right = right_cropped is not None
        if has_right:
            right_h, right_w = right_cropped.shape[:2]
            if (right_h, right_w) != (left_h, left_w):
                right_cropped = cv2.resize(right_cropped, (left_w, left_h))
        gap = 40
        label_margin = 30
        canvas_w = left_w if not has_right else left_w*2 + gap
        canvas_h = left_h + label_margin
        result_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        result_img.fill(50)
        result_img[label_margin:label_margin+left_h, :left_w] = left_cropped
        cv2.putText(result_img, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if has_right:
            x_off = left_w + gap
            result_img[label_margin:label_margin+left_h, x_off:x_off+left_w] = right_cropped
            cv2.putText(result_img, "RIGHT", (x_off+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        def draw_targets(w, img_w, x_off=0):
            for color, (tx_px, ty_px) in target_coords.items():
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                tx_cv = int(tx_bl) + x_off
                ty_cv = int(h_px - ty_bl) + label_margin
                color_map = {'C': (255, 255, 0), 'M': (255, 0, 255), 'Y': (0, 255, 255), 'S': (255, 255, 255)}
                draw_color = color_map.get(color, (255,255,255))
                cv2.line(result_img, (tx_cv-15, ty_cv-15), (tx_cv+15, ty_cv+15), draw_color, 4)
                cv2.line(result_img, (tx_cv-15, ty_cv+15), (tx_cv+15, ty_cv-15), draw_color, 4)
                cv2.line(result_img, (tx_cv-15, ty_cv-15), (tx_cv+15, ty_cv+15), (0,0,0), 2)
                cv2.line(result_img, (tx_cv-15, ty_cv+15), (tx_cv+15, ty_cv-15), (0,0,0), 2)
                mm_per_pixel_x = 5.0 / img_w
                mm_per_pixel_y = 5.0 / left_h
                tx_mm = tx_bl * mm_per_pixel_x
                ty_mm = ty_bl * mm_per_pixel_y
                cv2.putText(result_img, f"T{color}", (tx_cv+20, ty_cv), cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)
                cv2.putText(result_img, f"({tx_mm:.2f},{ty_mm:.2f})", (tx_cv+20, ty_cv+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        draw_targets(left_w, left_w, 0)
        if has_right:
            draw_targets(left_w, left_w, left_w + gap)
        
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
            
            # LATERAL - status와 관계없이 값이 있으면 표시
            lateral_item = QTableWidgetItem(f"{adj['LATERAL']:+.3f}" if adj.get('LATERAL') is not None else "N/A")
            lateral_item.setTextAlignment(Qt.AlignCenter)
            if adj.get('LATERAL') is not None:
                if adj['LATERAL'] > 0:
                    lateral_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                else:
                    lateral_item.setBackground(QColor(235, 255, 235))  # Light green for negative
            
            # CIRCUM - status와 관계없이 값이 있으면 표시
            circum_item = QTableWidgetItem(f"{adj['CIRCUM']:+.3f}" if adj.get('CIRCUM') is not None else "N/A")
            circum_item.setTextAlignment(Qt.AlignCenter)
            if adj.get('CIRCUM') is not None:
                if adj['CIRCUM'] > 0:
                    circum_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                else:
                    circum_item.setBackground(QColor(235, 255, 235))  # Light green for negative
            
            # COCKING - status와 관계없이 값이 있으면 표시
            cocking_item = QTableWidgetItem(f"{adj['COCKING']:+.3f}" if adj.get('COCKING') is not None else "N/A")
            cocking_item.setTextAlignment(Qt.AlignCenter)
            if adj.get('COCKING') is not None:
                if adj['COCKING'] > 0:
                    cocking_item.setBackground(QColor(255, 235, 235))  # Light red for positive
                else:
                    cocking_item.setBackground(QColor(235, 255, 235))  # Light green for negative
            
            # Status - 실제 status 값 표시
            status_text = adj.get('status', 'Unknown')
            if status_text == 'OK':
                status_item = QTableWidgetItem("✅ OK")
                status_item.setBackground(QColor(235, 255, 235))  # Light green
            elif status_text == 'Cocking data missing':
                status_item = QTableWidgetItem("⚠️ Cocking Missing")
                status_item.setBackground(QColor(255, 255, 200))  # Light yellow
            elif status_text == 'Warning':
                status_item = QTableWidgetItem("⚠️ Warning")
                status_item.setBackground(QColor(255, 235, 200))  # Light orange
            else:
                status_item = QTableWidgetItem(f"❌ {status_text}")
                status_item.setBackground(QColor(255, 220, 220))  # Light red
            
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