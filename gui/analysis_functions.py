#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis functions for CMYK color detection and image processing
Contains functions for special color detection, color classification, and square marker extraction
"""

import cv2
import numpy as np
from color_registration_analysis import (
    detect_bottom_left, detect_square_corners,
    pixel_to_bottom_left_coord, order_points
)

from .k_frame_detector import detect_k_frame_box
from .k_frame_face_detection import FACE_POSITION_KEYS


def detect_special_color(img: np.ndarray, exclude_colors: dict) -> tuple:
    """
    Detect special color box by first detecting all box shapes, then finding special color box.
    Identify special color by checking if C, M, Y, S boxes are all the same size.
    
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
        print(f"Warning: Could not find enough C, M, Y boxes. Found: {list(known_boxes.keys())}")
        # Continue with found boxes if any
        if len(known_boxes) == 0:
            print("Error: No CMY boxes found at all. Cannot continue.")
            return None, None
        print(f"Continuing with found boxes: {list(known_boxes.keys())}")
    
    # Step 2: Calculate average size of known boxes
    known_areas = [box['area'] for box in known_boxes.values()]
    avg_area = np.mean(known_areas)
    area_tolerance = avg_area * 0.3  # 30% tolerance
    
    print(f"Average area of known boxes: {avg_area:.0f} (Tolerance: ±{area_tolerance:.0f})")
    
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
                            print(f"Special candidate too close to {known_box['color']} box (distance: {distance:.1f})")
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
                                print(f"Special candidate overlaps {overlap_ratio:.1%} with {known_color} color")
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
        print("Could not find suitable special color box.")
        return None, None
    
    # Select box with most similar area
    best_box = min(potential_special_boxes, 
                   key=lambda x: abs(x['area'] - avg_area))
    
    print(f"Special color box found: Area {best_box['area']:.0f}, Center {best_box['center']}")
    
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
    Accurate color classification using Lab color space.
    
    Args:
        img: BGR image
        contour: Contour of color box
        
    Returns:
        str: Classified color ('C', 'M', 'Y', 'K')
    """
    # Create ROI mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    # Convert to Lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    
    # Calculate mean color within ROI
    mean_L = cv2.mean(L, mask=mask)[0]
    mean_a = cv2.mean(a, mask=mask)[0]
    mean_b = cv2.mean(b, mask=mask)[0]
    
    print(f"Lab values: L={mean_L:.1f}, a={mean_a:.1f}, b={mean_b:.1f}")
    
    # Classification based on Lab values
    # L: Brightness (0=black, 100=white)
    # a: Red-green (negative=green, positive=red)  
    # b: Blue-yellow (negative=blue, positive=yellow)
    
    if mean_L < 50:  # Dark color
        return 'S'  # Special color (black)
    elif mean_a > 20 and mean_b < 10:  # High red component, low yellow component
        return 'M'  # Magenta
    elif mean_b > 20 and mean_a < 10:  # High yellow component, low red component
        return 'Y'  # Yellow
    elif mean_a < -10 and mean_b < -10:  # Green/blue component
        return 'C'  # Cyan
    else:
        # Ambiguous case - use RGB-based secondary judgment
        mean_bgr = cv2.mean(img, mask=mask)[:3]
        b_val, g_val, r_val = mean_bgr
        
        if r_val > g_val and r_val > b_val:
            return 'M'  # Red series
        elif g_val > r_val and g_val > b_val:
            return 'C'  # Green/cyan series
        elif b_val > r_val and b_val > g_val:
            return 'C'  # Blue series
        else:
            return 'Y'  # Yellow series


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


def analyze_global_shift_pattern(registration_results):
    """
    Analyze global shift pattern to determine if all colors are shifted together.
    K (frame) movement means moving the entire frame that contains all color boxes.
    
    Args:
        registration_results: Color registration analysis results
        
    Returns:
        dict: Analysis results with shift pattern information
    """
    movements = []
    valid_colors = []
    actual_positions = []  # Actual positions of color boxes
    target_positions = []  # Target positions of color boxes
    
    for color in FACE_POSITION_KEYS:
        reg_data = registration_results.get(color)
        if reg_data and 'movement_mm' in reg_data:
            dx_mm, dy_mm = reg_data['movement_mm']
            movements.append((dx_mm, dy_mm))
            valid_colors.append(color)
            
            # Get actual and target positions
            if 'pivot_coord_mm' in reg_data and 'T_coord_mm' in reg_data:
                actual_positions.append(reg_data['pivot_coord_mm'])
                target_positions.append(reg_data['T_coord_mm'])
    
    if len(movements) < 2:
        return {
            'has_global_shift': False,
            'common_shift': (0.0, 0.0),
            'shift_consistency': 0.0,
            'strategy': 'individual',
            'frame_center_shift': (0.0, 0.0)
        }
    
    movements_array = np.array(movements)
    
    # Calculate frame center shift (K movement)
    # Average of all actual positions vs average of all target positions
    if actual_positions and target_positions:
        actual_center = np.mean(np.array(actual_positions), axis=0)
        target_center = np.mean(np.array(target_positions), axis=0)
        frame_center_shift = target_center - actual_center  # K frame movement
    else:
        frame_center_shift = np.array([0.0, 0.0])
    
    # Calculate median shift (more robust than mean)
    median_shift = np.median(movements_array, axis=0)
    
    # Calculate mean shift
    mean_shift = np.mean(movements_array, axis=0)
    
    # Use median for common shift (more robust to outliers)
    common_shift = median_shift
    
    # Calculate consistency: standard deviation of movements
    deviations = movements_array - common_shift
    distances = np.linalg.norm(deviations, axis=1)
    std_dev = np.std(distances)
    mean_distance = np.mean(distances)
    
    # Determine if there's a global shift pattern
    # If mean deviation is small compared to mean movement, it's a global shift
    mean_movement_magnitude = np.mean(np.linalg.norm(movements_array, axis=1))
    
    # Threshold: if deviations are less than 30% of mean movement, consider it global shift
    consistency_ratio = 1.0 - (mean_distance / (mean_movement_magnitude + 1e-6))
    has_global_shift = consistency_ratio > 0.7 and mean_movement_magnitude > 0.05
    
    print(f"Global shift analysis (K frame movement):")
    print(f"  Frame center shift (K): ({frame_center_shift[0]:.3f}, {frame_center_shift[1]:.3f}) mm")
    print(f"  Mean movement magnitude: {mean_movement_magnitude:.3f} mm")
    print(f"  Mean deviation from common shift: {mean_distance:.3f} mm")
    print(f"  Consistency ratio: {consistency_ratio:.2f}")
    print(f"  Common shift (median): ({common_shift[0]:.3f}, {common_shift[1]:.3f}) mm")
    print(f"  Has global shift: {has_global_shift}")
    
    return {
        'has_global_shift': has_global_shift,
        'common_shift': tuple(common_shift.tolist()),
        'mean_shift': tuple(mean_shift.tolist()),
        'frame_center_shift': tuple(frame_center_shift.tolist()),  # K frame movement
        'shift_consistency': consistency_ratio,
        'mean_deviation': float(mean_distance),
        'valid_colors': valid_colors,
        'strategy': 'global' if has_global_shift else 'individual'
    }


def find_optimal_common_shift(registration_results):
    """
    Find optimal common shift that minimizes total movement cost.
    K (frame) movement means moving the entire frame, so all colors move together.
    
    Args:
        registration_results: Color registration analysis results
        
    Returns:
        tuple: (optimal_common_shift_dx, optimal_common_shift_dy, total_cost)
    """
    movements = {}
    actual_positions = []
    target_positions = []
    
    for color in FACE_POSITION_KEYS:
        reg_data = registration_results.get(color)
        if reg_data and 'movement_mm' in reg_data:
            movements[color] = reg_data['movement_mm']
            if 'pivot_coord_mm' in reg_data and 'T_coord_mm' in reg_data:
                actual_positions.append(reg_data['pivot_coord_mm'])
                target_positions.append(reg_data['T_coord_mm'])
    
    if len(movements) < 2:
        return (0.0, 0.0), float('inf')
    
    # Calculate frame center shift (K movement)
    # This is the movement of the frame that contains all color boxes
    if actual_positions and target_positions:
        actual_center = np.mean(np.array(actual_positions), axis=0)
        target_center = np.mean(np.array(target_positions), axis=0)
        frame_center_shift = target_center - actual_center
    else:
        movements_array = np.array(list(movements.values()))
        frame_center_shift = np.mean(movements_array, axis=0)
    
    # Try different common shift values around frame center shift
    candidates = [
        (frame_center_shift[0], frame_center_shift[1])  # Frame center shift (K movement)
    ]
    
    # Also try median and mean of individual movements
    movements_array = np.array(list(movements.values()))
    median_shift = np.median(movements_array, axis=0)
    mean_shift = np.mean(movements_array, axis=0)
    candidates.extend([
        (median_shift[0], median_shift[1]),
        (mean_shift[0], mean_shift[1])
    ])
    
    # Test shifts in a small range around frame center
    for dx_offset in [-0.1, -0.05, 0.05, 0.1]:
        for dy_offset in [-0.1, -0.05, 0.05, 0.1]:
            candidates.append((frame_center_shift[0] + dx_offset, frame_center_shift[1] + dy_offset))
    
    best_cost = float('inf')
    best_shift = (0.0, 0.0)
    
    for common_dx, common_dy in candidates:
        total_cost = 0.0
        # Cost = |K_frame_shift| + sum of |residual| for each color
        # K frame shift moves all colors together, so cost is just |frame_shift|
        total_cost += abs(common_dx) + abs(common_dy)  # K frame movement cost
        
        for color, (dx, dy) in movements.items():
            residual_dx = dx - common_dx
            residual_dy = dy - common_dy
            # Residual movement cost (fine-tuning after K frame shift)
            total_cost += abs(residual_dx) + abs(residual_dy)
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_shift = (common_dx, common_dy)
    
    print(f"Optimal K frame shift: ({best_shift[0]:.3f}, {best_shift[1]:.3f}) mm, cost: {best_cost:.3f}")
    
    return best_shift, best_cost


def calculate_strategy_cost(registration_results, strategy_type, common_shift=None):
    """
    Calculate total cost for a given strategy.
    Cost = sum of absolute movement values for all colors (including K/S).
    
    Args:
        registration_results: Color registration analysis results
        strategy_type: 'global' or 'individual'
        common_shift: Common shift vector (dx, dy) for global strategy
        
    Returns:
        float: Total cost
    """
    if strategy_type == 'global' and common_shift is not None:
        # Global strategy: K (frame) moves, so all colors move together
        # Cost = |K_frame_shift| + sum of |residual| for each color
        dx, dy = common_shift
        total_cost = abs(dx) + abs(dy)  # K frame movement cost (moves entire frame)
        
        # Add residual costs for fine-tuning (individual color adjustments after frame shift)
        for color in FACE_POSITION_KEYS:
            reg_data = registration_results.get(color)
            if reg_data and 'movement_mm' in reg_data:
                color_dx, color_dy = reg_data['movement_mm']
                residual_dx = color_dx - dx
                residual_dy = color_dy - dy
                total_cost += abs(residual_dx) + abs(residual_dy)
        
        return total_cost
    
    elif strategy_type == 'individual':
        # Individual strategy: each color moves to its target (including K/S)
        # Cost = sum of |movement| for each color
        total_cost = 0.0
        for color in FACE_POSITION_KEYS:
            reg_data = registration_results.get(color)
            if reg_data and 'movement_mm' in reg_data:
                dx, dy = reg_data['movement_mm']
                total_cost += abs(dx) + abs(dy)
        return total_cost
    
    return float('inf')


def calculate_adjustment_values(registration_results, cocking_results=None):
    """
    Calculate LATERAL, CIRCUM, COCKING adjustment values for each color.
    Automatically determines optimal strategy: global shift vs individual correction.
    
    LATERAL: Left-right movement (X-axis)
    CIRCUM: Up-down movement (Y-axis) 
    COCKING: Y coordinate difference between left and right images
    
    Args:
        registration_results: Color registration analysis results
        cocking_results: Cocking analysis results (optional)
        
    Returns:
        dict: Adjustment values for each color with strategy information
    """
    print(f"DEBUG: registration_results keys: {list(registration_results.keys())}")
    
    # Step 1: Analyze global shift pattern
    shift_analysis = analyze_global_shift_pattern(registration_results)
    
    # Step 2: Find optimal common shift that minimizes total movement cost
    optimal_shift, optimal_global_cost = find_optimal_common_shift(registration_results)
    
    # Step 3: Calculate costs for both strategies
    global_cost = calculate_strategy_cost(registration_results, 'global', optimal_shift)
    individual_cost = calculate_strategy_cost(registration_results, 'individual')
    
    print(f"Strategy comparison:")
    print(f"  Optimal common shift: ({optimal_shift[0]:.3f}, {optimal_shift[1]:.3f}) mm")
    print(f"  Global strategy cost (with K/S): {global_cost:.3f}")
    print(f"  Individual strategy cost: {individual_cost:.3f}")
    
    # Step 4: Choose optimal strategy
    # Use global strategy if it reduces total movement cost
    use_global = global_cost < individual_cost * 0.95  # 5% margin for preference
    
    if use_global:
        print(f"  Selected strategy: GLOBAL (common shift: {optimal_shift}, total cost: {global_cost:.3f})")
        strategy = 'global'
        common_shift = optimal_shift
    else:
        print(f"  Selected strategy: INDIVIDUAL (total cost: {individual_cost:.3f})")
        strategy = 'individual'
        common_shift = None
    
    # Step 4: Calculate adjustments based on selected strategy
    adjustments = {}
    
    # Add K frame adjustment (if global strategy is used)
    if strategy == 'global' and common_shift is not None:
        k_dx, k_dy = common_shift
        # K frame movement direction should be opposite to the average movement needed
        # If all colors need to move left (negative), K frame should move right (positive) to compensate
        # common_shift is the average movement needed by colors
        # K adjustment = -common_shift (opposite direction)
        k_adjust_dx = -k_dx
        k_adjust_dy = -k_dy
        adjustments['K'] = {
            'LATERAL': round(k_adjust_dx, 3),  # Opposite to common_shift
            'CIRCUM': round(k_adjust_dy, 3),   # Opposite to common_shift
            'COCKING': None,  # K frame doesn't have cocking (it's the reference frame)
            'status': 'OK',
            'strategy': strategy,
            'is_frame': True
        }
        print(f"DEBUG: K - Frame shift: LATERAL={k_adjust_dx:.3f} (opposite of common {k_dx:.3f}), CIRCUM={k_adjust_dy:.3f} (opposite of common {k_dy:.3f})")
    else:
        # Individual strategy: K frame doesn't move separately
        adjustments['K'] = {
            'LATERAL': 0.0,
            'CIRCUM': 0.0,
            'COCKING': None,
            'status': 'N/A (Individual mode)',
            'strategy': strategy,
            'is_frame': True
        }
    
    for color in FACE_POSITION_KEYS:
        reg_data = registration_results.get(color)
        cocking_data = cocking_results.get(color) if cocking_results else None
        
        if reg_data is None:
            adjustments[color] = {
                'LATERAL': None,
                'CIRCUM': None,
                'COCKING': None,
                'status': 'Detection failed',
                'strategy': strategy
            }
            continue
            
        if 'movement_mm' not in reg_data:
            adjustments[color] = {
                'LATERAL': None,
                'CIRCUM': None,
                'COCKING': None,
                'status': 'Missing movement data',
                'strategy': strategy
            }
            continue
        
        dx_mm, dy_mm = reg_data['movement_mm']
        
        if strategy == 'global' and common_shift is not None:
            # Global strategy: K (frame) moves first, then each color fine-tunes
            # 
            # K frame = paper, Colors = ink positions
            # When K (paper) moves right (+), ink positions RELATIVELY move left (-) on the paper
            # 
            # K frame adjustment = -common_shift (opposite direction)
            # Relative effect on colors = opposite of K movement = +common_shift
            # 
            # Residual = original_movement_needed - relative_movement_from_K
            #          = dx_mm - (-k_adjust_dx) = dx_mm + k_adjust_dx
            common_dx, common_dy = common_shift
            k_adjust_dx = -common_dx  # K frame adjustment (opposite to common_shift)
            k_adjust_dy = -common_dy
            
            # When K moves right (+k_adjust), colors RELATIVELY move left (-k_adjust) on paper
            # So the relative effect is -k_adjust, and residual = dx_mm - (-k_adjust) = dx_mm + k_adjust
            residual_dx = dx_mm + k_adjust_dx  # Remaining adjustment after K's relative effect
            residual_dy = dy_mm + k_adjust_dy
            
            # In global strategy: K frame moves (affects all colors), then each color fine-tunes
            # The adjustment value shown to user should be: residual only (not total)
            # Because K movement already handles the common shift
            lateral_mm = residual_dx  # Only residual movement needed
            circum_mm = residual_dy
            
            residual_magnitude = np.sqrt(residual_dx**2 + residual_dy**2)
            if residual_magnitude < 0.02:  # Very small residual
                print(f"  {color}: K frame shift ({k_adjust_dx:.3f}, {k_adjust_dy:.3f}) covers most movement, residual: ({residual_dx:.3f}, {residual_dy:.3f})")
            else:
                print(f"  {color}: K frame shift ({k_adjust_dx:.3f}, {k_adjust_dy:.3f}) + color residual ({residual_dx:.3f}, {residual_dy:.3f}) = total needed ({dx_mm:.3f}, {dy_mm:.3f})")
        else:
            # Individual strategy: each color moves to its target independently
            lateral_mm = dx_mm
            circum_mm = dy_mm
        
        # COCKING = Y coordinate difference (right - left)
        cocking_mm = None
        if cocking_data and isinstance(cocking_data, dict) and 'cocking_mm' in cocking_data:
            cocking_mm = cocking_data['cocking_mm']
        
        # Determine status
        status = 'OK'
        if cocking_mm is None:
            status = 'Cocking data missing'
        elif abs(cocking_mm) > 0.5:
            status = 'Warning'
        
        adjustments[color] = {
            'LATERAL': round(lateral_mm, 3),
            'CIRCUM': round(circum_mm, 3),
            'COCKING': round(cocking_mm, 3) if cocking_mm is not None else None,
            'status': status,
            'strategy': strategy,
            'common_shift': common_shift if strategy == 'global' else None
        }
        
        print(f"DEBUG: {color} - Final adjustments: LATERAL={adjustments[color]['LATERAL']}, CIRCUM={adjustments[color]['CIRCUM']}, Strategy={strategy}")
    
    # Add strategy info to result
    adjustments['_strategy_info'] = {
        'strategy': strategy,
        'global_cost': global_cost,
        'individual_cost': individual_cost,
        'optimal_common_shift': optimal_shift if strategy == 'global' else None,
        'shift_analysis': shift_analysis
    }
    
    return adjustments

