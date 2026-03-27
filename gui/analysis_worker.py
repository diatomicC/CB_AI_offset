#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis worker thread for CMYK color registration analysis
Handles background processing of images and generates analysis results
"""

import cv2
import numpy as np
from datetime import datetime
from PySide6.QtCore import QThread, Signal

from color_registration_analysis import pixel_to_bottom_left_coord
from gui.analysis_functions import calculate_adjustment_values, detect_k_frame_box
from gui.k_frame_face_detection import (
    FACE_POSITION_KEYS,
    assign_face_boxes_to_positions,
    detect_color_faces_in_warp,
    pivot_point_from_face_quad,
)

K_FRAME_NORMALIZED_SIZE = 512


def _crop_to_k_frame(img: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perspective-warp image to the detected K-frame square.
    Returns:
        (warped_image, warped_k_corners_in_image_coords)
    """
    pts = corners.astype(np.float32)
    side = int(K_FRAME_NORMALIZED_SIZE)

    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (side, side))
    warped_corners = dst.copy()
    return warped, warped_corners


class AnalysisWorker(QThread):
    """Worker thread that executes analysis tasks in the background"""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, left_image_path, right_image_path=None):
        super().__init__()
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        self.special_color_name = "N/A (position-based faces)"
    
    def run(self):
        try:
            self.progress.emit("Loading image...")
            
            # Load left image
            left_orig = cv2.imread(self.left_image_path)
            if left_orig is None:
                self.error.emit("Cannot read left image file.")
                return
                
            self.progress.emit("Preparing image for K-frame detection...")
            
            # Step 1: Use full original image for detection
            preprocessed = left_orig.copy()
                
            self.progress.emit("Detecting K frame box...")
            
            # Step 2: Detect K frame box, then crop/warp to K-frame
            k_frame_corners = detect_k_frame_box(preprocessed)
            if k_frame_corners is None:
                self.error.emit("Cannot find K frame box.")
                return
            preprocessed, k_frame_corners = _crop_to_k_frame(preprocessed, k_frame_corners)
            
            # Calculate K frame center and size
            k_frame_center = np.mean(k_frame_corners, axis=0)
            k_frame_width = np.linalg.norm(k_frame_corners[1] - k_frame_corners[0])
            k_frame_height = np.linalg.norm(k_frame_corners[2] - k_frame_corners[1])
            k_frame_size = max(k_frame_width, k_frame_height)
            
            # Calculate mm per pixel based on K frame (assuming 5mm x 5mm)
            mm_per_pixel_x = 5.0 / k_frame_size
            mm_per_pixel_y = 5.0 / k_frame_size
            
            h_px, w_px = preprocessed.shape[:2]

            k_center_x, k_center_y = k_frame_center[0], k_frame_center[1]
            k_half_size = k_frame_size / 2
            target_coords = {
                "upper_left": (k_center_x - k_half_size * 0.2, k_center_y - k_half_size * 0.2),
                "upper_right": (k_center_x + k_half_size * 0.2, k_center_y - k_half_size * 0.2),
                "lower_left": (k_center_x - k_half_size * 0.2, k_center_y + k_half_size * 0.2),
                "lower_right": (k_center_x + k_half_size * 0.2, k_center_y + k_half_size * 0.2),
            }

            self.progress.emit("Detecting colored faces in K-frame...")
            faces = detect_color_faces_in_warp(preprocessed)
            pos_to_box = assign_face_boxes_to_positions(faces, w_px, h_px)

            self.progress.emit("Analyzing registration (position-based)...")

            def _is_inside_k_frame(x: float, y: float, k_corners: np.ndarray) -> bool:
                k_corners_int = k_corners.astype(np.int32)
                result = cv2.pointPolygonTest(k_corners_int, (float(x), float(y)), False)
                return result >= 0

            _POS_TAG = {
                "upper_left": "UL",
                "upper_right": "UR",
                "lower_left": "LL",
                "lower_right": "LR",
            }

            results_reg = {}
            debug_reg = preprocessed.copy()
            k_frame_int = k_frame_corners.astype(np.int32)
            cv2.polylines(debug_reg, [k_frame_int], True, (128, 128, 128), 2)

            for _pn, (tx_px, ty_px) in target_coords.items():
                tx_cv = int(tx_px)
                ty_cv = int(ty_px)
                cv2.line(debug_reg, (tx_cv - 10, ty_cv - 10), (tx_cv + 10, ty_cv + 10), (0, 0, 255), 2)
                cv2.line(debug_reg, (tx_cv - 10, ty_cv + 10), (tx_cv + 10, ty_cv - 10), (0, 0, 255), 2)
                cv2.putText(
                    debug_reg,
                    f"T{_POS_TAG[_pn]}",
                    (tx_cv + 12, ty_cv),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            for pos_name in FACE_POSITION_KEYS:
                box = pos_to_box.get(pos_name)
                if box is None:
                    print(f"{pos_name}: face detection failed")
                    results_reg[pos_name] = None
                    continue

                px_px, py_px = pivot_point_from_face_quad(box, pos_name)
                is_inside_k_frame = _is_inside_k_frame(px_px, py_px, k_frame_corners)

                px_bl, py_bl = pixel_to_bottom_left_coord(px_px, py_px, h_px)
                tx_px, ty_px = target_coords[pos_name]
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                dx_px = tx_bl - px_bl
                dy_px = ty_bl - py_bl
                px_mm = px_bl * mm_per_pixel_x
                py_mm = py_bl * mm_per_pixel_y
                tx_mm = tx_bl * mm_per_pixel_x
                ty_mm = ty_bl * mm_per_pixel_y
                dx_mm = dx_px * mm_per_pixel_x
                dy_mm = dy_px * mm_per_pixel_y

                print(f"DEBUG: {pos_name} - pivot({px_px:.1f}, {py_px:.1f}) -> T({tx_px:.1f}, {ty_px:.1f})")

                results_reg[pos_name] = {
                    "pivot_coord_mm": (round(px_mm, 3), round(py_mm, 3)),
                    "T_coord_mm": (round(tx_mm, 3), round(ty_mm, 3)),
                    "movement_mm": (round(dx_mm, 3), round(dy_mm, 3)),
                    "pivot_coord_px": (px_px, py_px),
                    "is_inside_k_frame": is_inside_k_frame,
                    "face_quad": box.astype(np.float32),
                }

                px_int, py_int = int(round(px_px)), int(round(py_px))
                if is_inside_k_frame:
                    circle_color = (0, 255, 0)
                    text_color = (0, 255, 0)
                else:
                    circle_color = (0, 165, 255)
                    text_color = (0, 165, 255)
                cv2.circle(debug_reg, (px_int, py_int), 8, circle_color, -1)
                cv2.circle(debug_reg, (px_int, py_int), 10, (0, 0, 0), 2)
                pts = box.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(debug_reg, [pts], True, text_color, 2)
                cv2.putText(
                    debug_reg,
                    f"Pv{_POS_TAG[pos_name]}({px_mm:.2f},{py_mm:.2f})",
                    (px_int + 15, py_int - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1,
                )
                cv2.putText(
                    debug_reg,
                    f"Move({dx_mm:.2f},{dy_mm:.2f})",
                    (px_int + 15, py_int + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                if not is_inside_k_frame:
                    tx_int, ty_int = int(tx_px), int(ty_px)
                    cv2.arrowedLine(
                        debug_reg, (px_int, py_int), (tx_int, ty_int), (255, 0, 255), 2, tipLength=0.02
                    )

            self.progress.emit("Analysis completed!")
            
            # Analyze right image for cocking calculation if available
            cocking_results = {}
            cocking_debug_img = None
            right_results_reg = {}
            right_preprocessed = None
            right_k_frame_corners = None
            
            if self.right_image_path:
                self.progress.emit("Analyzing right image for cocking calculation...")
                
                right_orig = cv2.imread(self.right_image_path)
                if right_orig is not None:
                    # Use full original image for right side detection, then crop to K-frame
                    right_preprocessed = right_orig.copy()
                    if right_preprocessed is not None:
                        right_k_frame_corners = detect_k_frame_box(right_preprocessed)
                        if right_k_frame_corners is not None:
                            right_preprocessed, right_k_frame_corners = _crop_to_k_frame(
                                right_preprocessed, right_k_frame_corners
                            )
                            # Calculate right K frame center and size
                            right_k_frame_center = np.mean(right_k_frame_corners, axis=0)
                            right_k_frame_width = np.linalg.norm(right_k_frame_corners[1] - right_k_frame_corners[0])
                            right_k_frame_height = np.linalg.norm(right_k_frame_corners[2] - right_k_frame_corners[1])
                            right_k_frame_size = max(right_k_frame_width, right_k_frame_height)
                            
                            right_h_px, right_w_px = right_preprocessed.shape[:2]
                            right_mm_per_pixel_x = 5.0 / right_k_frame_size
                            right_mm_per_pixel_y = 5.0 / right_k_frame_size
                            
                            right_k_center_x, right_k_center_y = right_k_frame_center[0], right_k_frame_center[1]
                            right_k_half_size = right_k_frame_size / 2
                            right_target_coords = {
                                "upper_left": (
                                    right_k_center_x - right_k_half_size * 0.2,
                                    right_k_center_y - right_k_half_size * 0.2,
                                ),
                                "upper_right": (
                                    right_k_center_x + right_k_half_size * 0.2,
                                    right_k_center_y - right_k_half_size * 0.2,
                                ),
                                "lower_left": (
                                    right_k_center_x - right_k_half_size * 0.2,
                                    right_k_center_y + right_k_half_size * 0.2,
                                ),
                                "lower_right": (
                                    right_k_center_x + right_k_half_size * 0.2,
                                    right_k_center_y + right_k_half_size * 0.2,
                                ),
                            }

                            right_faces = detect_color_faces_in_warp(right_preprocessed)
                            right_pos_to_box = assign_face_boxes_to_positions(
                                right_faces, right_w_px, right_h_px
                            )

                            for pos_name in FACE_POSITION_KEYS:
                                box = right_pos_to_box.get(pos_name)
                                if box is None:
                                    print(f"RIGHT {pos_name}: face detection failed")
                                    right_results_reg[pos_name] = None
                                    continue

                                px_px, py_px = pivot_point_from_face_quad(box, pos_name)
                                px_bl, py_bl = pixel_to_bottom_left_coord(px_px, py_px, right_h_px)
                                tx_px, ty_px = right_target_coords[pos_name]
                                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, right_h_px)
                                right_results_reg[pos_name] = {
                                    "pivot_coord_mm": (
                                        px_bl * right_mm_per_pixel_x,
                                        py_bl * right_mm_per_pixel_y,
                                    ),
                                    "T_coord_mm": (
                                        tx_bl * right_mm_per_pixel_x,
                                        ty_bl * right_mm_per_pixel_y,
                                    ),
                                    "pivot_coord_px": (px_px, py_px),
                                    "face_quad": box.astype(np.float32),
                                }
                            
                            # Calculate cocking using both left and right results
                            cocking_results, cocking_debug_img = self.calculate_cocking_from_results(
                                results_reg, right_results_reg, preprocessed, right_preprocessed, mm_per_pixel_y
                            )
            
            # Calculate adjustment values (including K frame movement decision)
            # calculate_adjustment_values already computes:
            # - K adjustment (if global strategy)
            # - Color residuals (remaining movement after K shift)
            # No need to recalculate here - the values are already correct!
            adjustments = calculate_adjustment_values(results_reg, cocking_results)
            
            # Log the strategy and K adjustment for debugging
            k_adj = adjustments.get('K', {})
            k_lateral = k_adj.get('LATERAL', 0.0)
            k_circum = k_adj.get('CIRCUM', 0.0)
            strategy = k_adj.get('strategy', 'individual')
            
            if strategy == 'global' and (k_lateral != 0 or k_circum != 0):
                print(f"DEBUG: Global strategy - K frame adjustment: LATERAL={k_lateral:.3f}, CIRCUM={k_circum:.3f}")
                print(f"DEBUG: Color adjustments are residuals after K frame movement")
                for pos_name in FACE_POSITION_KEYS:
                    if pos_name in adjustments and adjustments[pos_name]:
                        adj = adjustments[pos_name]
                        if adj.get('LATERAL') is not None:
                            print(f"  {pos_name}: LATERAL={adj['LATERAL']:.3f}, CIRCUM={adj['CIRCUM']:.3f}")
            else:
                print(f"DEBUG: Individual strategy - Each color moves independently")
            
            # Create visualization images
            preprocessed_img = self.create_preprocessed_image(preprocessed, right_preprocessed)
            cmyk_detection_img = self.create_face_detection_image(preprocessed, right_preprocessed, results_reg, right_results_reg)
            pivot_img = self.create_pivot_image(preprocessed, right_preprocessed, results_reg, right_results_reg, h_px)
            t_points_img = self.create_t_points_image(preprocessed, right_preprocessed, target_coords, h_px)
            
            # Create after adjustment visualization (use full original image, not cropped)
            right_orig_var = None
            if self.right_image_path:
                right_orig_var = cv2.imread(self.right_image_path) if 'right_orig' not in locals() else right_orig
            after_adjustment_img = self.create_after_adjustment_image(
                left_orig, right_orig_var, 
                results_reg, adjustments, 
                k_frame_corners, target_coords, h_px, mm_per_pixel_x, mm_per_pixel_y,
                preprocessed  # Pass preprocessed for coordinate reference
            )
            
            # Return results
            results = {
                'registration': results_reg,
                'cocking': cocking_results,
                'adjustments': adjustments,
                'debug_reg': debug_reg,
                'cocking_debug': cocking_debug_img,
                'preprocessed': preprocessed_img,
                'cmyk_detection': cmyk_detection_img,
                'pivot': pivot_img,
                't_points': t_points_img,
                'after_adjustment': after_adjustment_img,
                'special_color_name': self.special_color_name,
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
            print(f"Analysis error: {str(e)}")
            if 'results' in locals() and results:
                print("Returning partial results despite error...")
                self.finished.emit(results)
            else:
                self.error.emit(f"Critical error during analysis: {str(e)}")
    
    def calculate_cocking_from_results(self, left_results_reg, right_results_reg, left_preprocessed, right_preprocessed, mm_per_pixel_y):
        """Calculate cocking values by comparing Y coordinates between left and right images."""
        try:
            print("=" * 50)
            print("COCKING ALGORITHM - USING PRE-PROCESSED RESULTS")
            print("=" * 50)
            
            if not left_results_reg or not right_results_reg:
                print("ERROR: Missing left or right image results")
                return {}, None
            
            print(f"Left results: {list(left_results_reg.keys())}")
            print(f"Right results: {list(right_results_reg.keys())}")
            
            # Prepare left image results for comparison
            left_results_cocking = {}
            
            for pos_name, data in left_results_reg.items():
                if data and 'pivot_coord_px' in data:
                    px_px, py_px = data['pivot_coord_px']
                    left_results_cocking[pos_name] = {
                        'pivot_coord_px': [px_px, py_px],
                        'position': pos_name,
                    }
                    print(f"    {pos_name}: pivot at ({px_px:.1f}, {py_px:.1f}) px (left)")
                else:
                    print(f"    {pos_name}: No valid data from left image")
                    left_results_cocking[pos_name] = None
            
            # Calculate cocking for each color
            cocking_results = {}
            
            print("\nCOCKING ANALYSIS RESULTS:")
            print("-" * 40)
            
            for pos_name in FACE_POSITION_KEYS:
                left_data = left_results_cocking.get(pos_name)
                right_data = right_results_reg.get(pos_name)
                
                if left_data and right_data and left_data is not None and right_data is not None:
                    try:
                        left_y_px = left_data['pivot_coord_px'][1]
                        right_y_px = right_data['pivot_coord_px'][1]
                        
                        y_diff_px = right_y_px - left_y_px
                        y_diff_mm = y_diff_px * mm_per_pixel_y
                        
                        cocking_results[pos_name] = {
                            'left_y_px': round(left_y_px, 2),
                            'right_y_px': round(right_y_px, 2),
                            'y_diff_px': round(y_diff_px, 2),
                            'cocking_mm': round(y_diff_mm, 3),
                            'status': 'OK'
                        }
                        
                        status_symbol = "^" if y_diff_mm > 0 else "v" if y_diff_mm < 0 else "-"
                        print(f"{pos_name}: Left={left_y_px:.1f}px, Right={right_y_px:.1f}px -> {status_symbol} {y_diff_mm:+.3f}mm")
                        
                    except Exception as e:
                        print(f"{pos_name}: Calculation failed - {str(e)}")
                        cocking_results[pos_name] = {
                            'status': 'Error',
                            'error': str(e)
                        }
                else:
                    missing = []
                    if not left_data or left_data is None:
                        missing.append("left")
                    if not right_data or right_data is None:
                        missing.append("right")
                    
                    print(f"{pos_name}: Missing data from {', '.join(missing)} image(s)")
                    cocking_results[pos_name] = {
                        'status': 'Missing data',
                        'missing': missing
                    }
            
            # Create debug visualization
            print("STEP 4: Creating cocking debug visualization...")
            debug_img = self.create_cocking_debug_image(
                left_preprocessed, right_preprocessed, 
                left_results_cocking, right_results_reg, 
                mm_per_pixel_y
            )
            
            print("=" * 50)
            print("COCKING ANALYSIS COMPLETED")
            print("=" * 50)
            
            return cocking_results, debug_img
            
        except Exception as e:
            print(f"CRITICAL ERROR in cocking calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}, None
    
    def create_cocking_debug_image(self, left_preprocessed, right_preprocessed, left_results, right_results, mm_per_pixel_y):
        """Create comprehensive debug visualization for cocking analysis."""
        try:
            print("Creating enhanced cocking debug visualization...")
            
            left_h, left_w = left_preprocessed.shape[:2]
            right_h, right_w = right_preprocessed.shape[:2]
            
            margin = 100
            max_h = max(left_h, right_h) + margin
            combined_w = left_w + right_w + 40
            
            combined_img = np.zeros((max_h, combined_w, 3), dtype=np.uint8)
            combined_img.fill(50)
            
            y_offset = margin // 2
            combined_img[y_offset:y_offset + left_h, :left_w] = left_preprocessed
            
            right_x_offset = left_w + 40
            combined_img[y_offset:y_offset + right_h, right_x_offset:right_x_offset + right_w] = right_preprocessed
            
            cv2.putText(combined_img, "LEFT IMAGE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined_img, "RIGHT IMAGE", (right_x_offset + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            colors_rgb = {
                'upper_left': (255, 0, 0),
                'upper_right': (255, 0, 255),
                'lower_left': (0, 165, 255),
                'lower_right': (0, 255, 255),
            }
            
            measurement_y = max_h - 60
            
            for i, pos_name in enumerate(FACE_POSITION_KEYS):
                left_data = left_results.get(pos_name)
                right_data = right_results.get(pos_name)
                color_rgb = colors_rgb.get(pos_name, (255, 255, 255))
                
                if left_data and right_data and left_data is not None and right_data is not None:
                    left_x, left_y = left_data['pivot_coord_px']
                    right_x, right_y = right_data['pivot_coord_px']
                    
                    left_display_x = int(left_x)
                    left_display_y = int(left_y + y_offset)
                    right_display_x = int(right_x + right_x_offset)
                    right_display_y = int(right_y + y_offset)
                    
                    cv2.circle(combined_img, (left_display_x, left_display_y), 6, color_rgb, -1)
                    cv2.circle(combined_img, (left_display_x, left_display_y), 8, (255, 255, 255), 2)
                    cv2.circle(combined_img, (right_display_x, right_display_y), 6, color_rgb, -1)
                    cv2.circle(combined_img, (right_display_x, right_display_y), 8, (255, 255, 255), 2)
                    
                    cv2.putText(combined_img, f"Pv{pos_name[:2]}", 
                               (left_display_x + 12, left_display_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
                    cv2.putText(combined_img, f"Pv{pos_name[:2]}", 
                               (right_display_x + 12, right_display_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
                    
                    line_y = (left_display_y + right_display_y) // 2
                    cv2.line(combined_img, (left_display_x, line_y), 
                            (right_display_x, line_y), color_rgb, 2)
                    
                    y_diff_px = right_y - left_y
                    y_diff_mm = y_diff_px * mm_per_pixel_y
                    
                    measurement_text = f"{pos_name[:8]}: {y_diff_mm:+.3f}mm"
                    text_x = 10 + (i * 120)
                    cv2.putText(combined_img, measurement_text, 
                               (text_x, measurement_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)
                    
                    mid_x = (left_display_x + right_display_x) // 2
                    mid_y = line_y
                    
                    if abs(y_diff_mm) > 0.001:
                        arrow_color = (0, 255, 0) if y_diff_mm > 0 else (0, 0, 255)
                        if y_diff_mm > 0:
                            cv2.arrowedLine(combined_img, (mid_x, mid_y - 10), (mid_x, mid_y + 10), arrow_color, 2)
                        else:
                            cv2.arrowedLine(combined_img, (mid_x, mid_y + 10), (mid_x, mid_y - 10), arrow_color, 2)
                else:
                    measurement_text = f"{pos_name[:8]}: N/A"
                    text_x = 10 + (i * 120)
                    cv2.putText(combined_img, measurement_text, 
                               (text_x, measurement_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            legend_y = max_h - 30
            cv2.putText(combined_img, "Green Arrow: Right Lower | Red Arrow: Right Higher", 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print("Cocking debug image created successfully")
            return combined_img
            
        except Exception as e:
            print(f"Error creating cocking debug image: {str(e)}")
            try:
                left_h, left_w = left_preprocessed.shape[:2]
                right_h, right_w = right_preprocessed.shape[:2]
                max_h = max(left_h, right_h)
                combined_img = np.zeros((max_h, left_w + right_w, 3), dtype=np.uint8)
                combined_img[:left_h, :left_w] = left_preprocessed
                combined_img[:right_h, left_w:] = right_preprocessed
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
        
        gap = 40
        label_margin = 30
        canvas_w = left_w if not has_right else left_w * 2 + gap
        canvas_h = left_h + label_margin
        result_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        result_img.fill(50)
        
        result_img[label_margin:label_margin+left_h, :left_w] = left_preprocessed
        cv2.putText(result_img, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        if has_right:
            x_off = left_w + gap
            result_img[label_margin:label_margin+left_h, x_off:x_off+left_w] = right_preprocessed
            cv2.putText(result_img, "RIGHT", (x_off+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        return result_img

    def create_face_detection_image(self, left_preprocessed, right_preprocessed, left_results=None, right_results=None):
        """Draw detected face quads (position-based) on warped K-frame images."""
        left_h, left_w = left_preprocessed.shape[:2]
        has_right = right_preprocessed is not None
        if has_right:
            right_h, right_w = right_preprocessed.shape[:2]
            if (right_h, right_w) != (left_h, left_w):
                right_preprocessed = cv2.resize(right_preprocessed, (left_w, left_h))

        gap = 40
        label_margin = 30
        canvas_w = left_w if not has_right else left_w * 2 + gap
        canvas_h = left_h + label_margin
        result_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        result_img.fill(50)

        result_img[label_margin : label_margin + left_h, :left_w] = left_preprocessed
        cv2.putText(result_img, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if has_right:
            x_off = left_w + gap
            result_img[label_margin : label_margin + left_h, x_off : x_off + left_w] = right_preprocessed
            cv2.putText(result_img, "RIGHT", (x_off + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        color_map = {
            "upper_left": (255, 0, 0),
            "upper_right": (255, 0, 255),
            "lower_left": (0, 165, 255),
            "lower_right": (0, 255, 255),
        }

        def draw_for(x_offset: int, results: dict | None) -> None:
            if not results:
                return
            for pos_name, reg_data in results.items():
                if not reg_data:
                    continue
                draw_color = color_map.get(pos_name, (255, 255, 255))
                if "face_quad" in reg_data and reg_data["face_quad"] is not None:
                    quad = reg_data["face_quad"].astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        result_img,
                        [quad + np.array([x_offset, label_margin])],
                        True,
                        draw_color,
                        2,
                    )
                px_px = py_px = None
                if "pivot_coord_px" in reg_data:
                    px_px, py_px = reg_data["pivot_coord_px"]
                if px_px is None:
                    continue
                px_cv = int(px_px) + x_offset
                py_cv = int(py_px) + label_margin
                short = pos_name.replace("_", " ")[:11]
                cv2.putText(
                    result_img,
                    short,
                    (px_cv - 10, py_cv),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    draw_color,
                    2,
                )

        draw_for(0, left_results)
        if has_right:
            draw_for(left_w + gap, right_results)

        return result_img
    
    def create_pivot_image(self, left_preprocessed, right_preprocessed, left_results_reg, right_results_reg, h_px):
        """Create image showing pivot points for left and right"""
        left_h, left_w = left_preprocessed.shape[:2]
        has_right = right_preprocessed is not None and bool(right_results_reg)
        if has_right:
            right_h, right_w = right_preprocessed.shape[:2]
            if (right_h, right_w) != (left_h, left_w):
                right_preprocessed = cv2.resize(right_preprocessed, (left_w, left_h))
        gap = 40
        label_margin = 30
        canvas_w = left_w if not has_right else left_w*2 + gap
        canvas_h = left_h + label_margin
        result_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        result_img.fill(50)
        result_img[label_margin:label_margin+left_h, :left_w] = left_preprocessed
        cv2.putText(result_img, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if has_right:
            x_off = left_w + gap
            result_img[label_margin:label_margin+left_h, x_off:x_off+left_w] = right_preprocessed
            cv2.putText(result_img, "RIGHT", (x_off+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        def draw_points(results, mm_w, mm_h, x_off=0):
            for pos_name, reg_data in results.items():
                if not reg_data:
                    continue
                px_mm, py_mm = reg_data['pivot_coord_mm']
                px_bl = px_mm / mm_w
                py_bl = py_mm / mm_h
                px_cv = int(px_bl) + x_off
                py_cv = int(h_px - py_bl)
                color_map = {
                    "upper_left": (255, 255, 0),
                    "upper_right": (255, 0, 255),
                    "lower_left": (0, 255, 255),
                    "lower_right": (255, 255, 255),
                }
                draw_color = color_map.get(pos_name, (255, 255, 255))
                cv2.circle(result_img, (px_cv, py_cv+label_margin), 12, draw_color, -1)
                cv2.circle(result_img, (px_cv, py_cv+label_margin), 15, (0,0,0), 3)
                cv2.putText(result_img, f"Pivot {pos_name}", (px_cv+20, py_cv+label_margin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
                cv2.putText(result_img, f"({px_mm:.2f},{py_mm:.2f})", (px_cv+20, py_cv+label_margin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Use K frame size for mm_per_pixel calculation (assuming 5mm x 5mm)
        mm_per_pixel_x_left = 5.0 / left_preprocessed.shape[1]  # Approximate
        mm_per_pixel_y_left = 5.0 / left_preprocessed.shape[0]  # Approximate
        draw_points(left_results_reg, mm_per_pixel_x_left, mm_per_pixel_y_left, 0)
        if has_right:
            mm_per_pixel_x_right = 5.0 / right_preprocessed.shape[1]  # Approximate
            mm_per_pixel_y_right = 5.0 / right_preprocessed.shape[0]  # Approximate
            draw_points(right_results_reg, mm_per_pixel_x_right, mm_per_pixel_y_right, left_w + gap)
        
        return result_img
    
    def create_t_points_image(self, left_preprocessed, right_preprocessed, target_coords, h_px):
        """Create image showing T points side-by-side if right available"""
        left_h, left_w = left_preprocessed.shape[:2]
        has_right = right_preprocessed is not None
        if has_right:
            right_h, right_w = right_preprocessed.shape[:2]
            if (right_h, right_w) != (left_h, left_w):
                right_preprocessed = cv2.resize(right_preprocessed, (left_w, left_h))
        gap = 40
        label_margin = 30
        canvas_w = left_w if not has_right else left_w*2 + gap
        canvas_h = left_h + label_margin
        result_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        result_img.fill(50)
        result_img[label_margin:label_margin+left_h, :left_w] = left_preprocessed
        cv2.putText(result_img, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if has_right:
            x_off = left_w + gap
            result_img[label_margin:label_margin+left_h, x_off:x_off+left_w] = right_preprocessed
            cv2.putText(result_img, "RIGHT", (x_off+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        def draw_targets(w, img_w, x_off=0):
            for pos_name, (tx_px, ty_px) in target_coords.items():
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                tx_cv = int(tx_bl) + x_off
                ty_cv = int(h_px - ty_bl) + label_margin
                color_map = {
                    "upper_left": (255, 255, 0),
                    "upper_right": (255, 0, 255),
                    "lower_left": (0, 255, 255),
                    "lower_right": (255, 255, 255),
                }
                draw_color = color_map.get(pos_name, (255, 255, 255))
                cv2.line(result_img, (tx_cv-15, ty_cv-15), (tx_cv+15, ty_cv+15), draw_color, 4)
                cv2.line(result_img, (tx_cv-15, ty_cv+15), (tx_cv+15, ty_cv-15), draw_color, 4)
                cv2.line(result_img, (tx_cv-15, ty_cv-15), (tx_cv+15, ty_cv+15), (0,0,0), 2)
                cv2.line(result_img, (tx_cv-15, ty_cv+15), (tx_cv+15, ty_cv-15), (0,0,0), 2)
                mm_per_pixel_x = 5.0 / img_w
                mm_per_pixel_y = 5.0 / left_h
                tx_mm = tx_bl * mm_per_pixel_x
                ty_mm = ty_bl * mm_per_pixel_y
                cv2.putText(result_img, f"T {pos_name}", (tx_cv+20, ty_cv), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)
                cv2.putText(result_img, f"({tx_mm:.2f},{ty_mm:.2f})", (tx_cv+20, ty_cv+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        draw_targets(left_w, left_w, 0)
        if has_right:
            draw_targets(left_w, left_w, left_w + gap)
        
        return result_img
    
    def create_after_adjustment_image(self, left_orig, right_orig, results_reg, adjustments, 
                                     k_frame_corners, target_coords, h_px, mm_per_pixel_x, mm_per_pixel_y,
                                     preprocessed=None):
        """Create visualization showing positions after applying adjustment values
        Uses full original image instead of cropped preprocessed image.
        Shows K frame movement and color positions relative to moved K frame.
        """
        try:
            if left_orig is None:
                return None
            
            # Use full original image instead of cropped preprocessed
            # If preprocessed is provided, we'll need to map coordinates
            # For now, use preprocessed if it's the same size, otherwise use original
            # In practice, extract_robust_square_marker crops the image, so we use original
            result_img = left_orig.copy()
            
            # Get image dimensions
            orig_h, orig_w = left_orig.shape[:2]
            if preprocessed is not None:
                prep_h, prep_w = preprocessed.shape[:2]
                # If preprocessed is smaller, we need to find the crop region in original
                # For simplicity, try to detect K frame in original image directly
                k_frame_in_orig = detect_k_frame_box(left_orig)
                if k_frame_in_orig is not None:
                    k_frame_corners_orig = k_frame_in_orig
                    # Calculate offset if needed
                    k_frame_center_prep = np.mean(k_frame_corners, axis=0)
                    k_frame_center_orig = np.mean(k_frame_in_orig, axis=0)
                    offset_x = k_frame_center_orig[0] - k_frame_center_prep[0]
                    offset_y = k_frame_center_orig[1] - k_frame_center_prep[1]
                else:
                    # Fallback: assume preprocessed is centered in original
                    k_frame_corners_orig = k_frame_corners.copy()
                    offset_x = (orig_w - prep_w) / 2
                    offset_y = (orig_h - prep_h) / 2
                    k_frame_corners_orig[:, 0] += offset_x
                    k_frame_corners_orig[:, 1] += offset_y
            else:
                k_frame_corners_orig = k_frame_corners.copy()
                offset_x = 0
                offset_y = 0
            
            # Calculate K frame adjustment
            k_adj = adjustments.get('K', {})
            k_lateral = k_adj.get('LATERAL', 0.0)
            k_circum = k_adj.get('CIRCUM', 0.0)
            strategy = k_adj.get('strategy', 'individual')
            
            # Calculate moved K frame position
            k_dx_px = k_lateral / mm_per_pixel_x if k_lateral else 0
            k_dy_px = -k_circum / mm_per_pixel_y if k_circum else 0  # Y is inverted
            moved_k_frame_corners = k_frame_corners_orig.copy()
            moved_k_frame_corners[:, 0] += k_dx_px
            moved_k_frame_corners[:, 1] += k_dy_px
            
            # Draw original K frame box (dashed/gray)
            k_frame_int = k_frame_corners_orig.astype(np.int32)
            cv2.polylines(result_img, [k_frame_int], True, (100, 100, 100), 2, cv2.LINE_AA)
            cv2.putText(result_img, "K Frame (Original)", (k_frame_int[0][0], k_frame_int[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # Draw moved K frame box (solid/white)
            moved_k_frame_int = moved_k_frame_corners.astype(np.int32)
            cv2.polylines(result_img, [moved_k_frame_int], True, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(result_img, "K Frame (After Adjustment)", (moved_k_frame_int[0][0], moved_k_frame_int[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate moved K frame center and size
            moved_k_center = np.mean(moved_k_frame_corners, axis=0)
            moved_k_frame_width = np.linalg.norm(moved_k_frame_corners[1] - moved_k_frame_corners[0])
            moved_k_frame_height = np.linalg.norm(moved_k_frame_corners[2] - moved_k_frame_corners[1])
            moved_k_frame_size = max(moved_k_frame_width, moved_k_frame_height)
            moved_k_half_size = moved_k_frame_size / 2
            
            # Recalculate target positions relative to MOVED K frame center
            # This is the key fix: targets should be relative to the moved K frame, not original
            moved_k_center_x, moved_k_center_y = moved_k_center[0], moved_k_center[1]
            moved_target_coords = {
                "upper_left": (
                    moved_k_center_x - moved_k_half_size * 0.2,
                    moved_k_center_y - moved_k_half_size * 0.2,
                ),
                "upper_right": (
                    moved_k_center_x + moved_k_half_size * 0.2,
                    moved_k_center_y - moved_k_half_size * 0.2,
                ),
                "lower_left": (
                    moved_k_center_x - moved_k_half_size * 0.2,
                    moved_k_center_y + moved_k_half_size * 0.2,
                ),
                "lower_right": (
                    moved_k_center_x + moved_k_half_size * 0.2,
                    moved_k_center_y + moved_k_half_size * 0.2,
                ),
            }

            color_map = {
                "upper_left": (255, 0, 0),
                "upper_right": (255, 0, 255),
                "lower_left": (0, 165, 255),
                "lower_right": (0, 255, 255),
                "K": (200, 200, 200),
            }

            for name in list(FACE_POSITION_KEYS) + ["K"]:
                adj_data = adjustments.get(name, {})

                if adj_data is None:
                    continue

                if name == "K":
                    k_center_orig = np.mean(k_frame_corners_orig, axis=0)
                    px_px, py_px = float(k_center_orig[0]), float(k_center_orig[1])
                else:
                    reg_data = results_reg.get(name)
                    if reg_data is None or "pivot_coord_px" not in reg_data:
                        continue
                    px_prep, py_prep = reg_data["pivot_coord_px"]
                    px_px = px_prep + offset_x
                    py_px = py_prep + offset_y

                px_int, py_int = int(px_px), int(py_px)

                lateral_mm = adj_data.get("LATERAL")
                circum_mm = adj_data.get("CIRCUM")

                if lateral_mm is None or circum_mm is None:
                    continue

                dx_px = lateral_mm / mm_per_pixel_x
                dy_px = circum_mm / mm_per_pixel_y

                new_x_px = px_px + dx_px
                new_y_px = py_px - dy_px
                new_x_int, new_y_int = int(new_x_px), int(new_y_px)

                if name == "K":
                    tx_px, ty_px = float(moved_k_center[0]), float(moved_k_center[1])
                else:
                    tx_px, ty_px = moved_target_coords.get(name, (px_px, py_px))

                tx_int, ty_int = int(tx_px), int(ty_px)

                distance_to_target = ((new_x_px - tx_px) ** 2 + (new_y_px - ty_px) ** 2) ** 0.5
                if distance_to_target > 2.0:
                    print(f"WARNING: {name} after adjustment is {distance_to_target:.1f}px from target!")
                    print(f"  After: ({new_x_px:.1f}, {new_y_px:.1f}), Target: ({tx_px:.1f}, {ty_px:.1f})")

                color_rgb = color_map.get(name, (255, 255, 255))

                cv2.circle(result_img, (px_int, py_int), 10, color_rgb, 2)
                cv2.putText(
                    result_img,
                    f"Pivot {name}",
                    (px_int + 15, py_int - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color_rgb,
                    2,
                )

                cv2.arrowedLine(
                    result_img, (px_int, py_int), (new_x_int, new_y_int), color_rgb, 2, tipLength=0.02
                )

                cv2.line(result_img, (tx_int - 8, ty_int - 8), (tx_int + 8, ty_int + 8), (0, 255, 0), 2)
                cv2.line(result_img, (tx_int - 8, ty_int + 8), (tx_int + 8, ty_int - 8), (0, 255, 0), 2)
                cv2.putText(
                    result_img,
                    f"T {name}",
                    (tx_int + 12, ty_int),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                cv2.circle(result_img, (new_x_int, new_y_int), 12, color_rgb, -1)
                cv2.circle(result_img, (new_x_int, new_y_int), 14, (255, 255, 255), 2)
                cv2.putText(
                    result_img,
                    f"After {name}",
                    (new_x_int + 15, new_y_int + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_rgb,
                    2,
                )
                
                # If After and Target are close enough, show success indicator
                if distance_to_target <= 2.0:
                    cv2.circle(result_img, (new_x_int, new_y_int), 16, (0, 255, 0), 1)  # Green outer ring
            
            # Add legend
            legend_y = 30
            legend_x = 10
            cv2.putText(result_img, "Legend:", (legend_x, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            legend_y += 25
            
            # Pivot: current position (hollow circle)
            cv2.circle(result_img, (legend_x + 10, legend_y), 8, (255, 255, 255), 2)
            cv2.putText(result_img, "Pivot = current position", (legend_x + 25, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 20
            
            # After: Position after applying adjustment (filled circle)
            cv2.circle(result_img, (legend_x + 10, legend_y), 10, (255, 255, 255), -1)
            cv2.circle(result_img, (legend_x + 10, legend_y), 12, (255, 255, 255), 2)
            cv2.putText(result_img, "After = Position after adjustment", (legend_x + 25, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 20
            
            # T: Target position (X mark)
            cv2.line(result_img, (legend_x + 2, legend_y - 6), (legend_x + 18, legend_y + 6), (0, 255, 0), 2)
            cv2.line(result_img, (legend_x + 2, legend_y + 6), (legend_x + 18, legend_y - 6), (0, 255, 0), 2)
            cv2.putText(result_img, "T = Target position", (legend_x + 25, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 20
            
            # Arrow: Shows adjustment movement
            cv2.arrowedLine(result_img, (legend_x + 5, legend_y), (legend_x + 20, legend_y), 
                          (255, 255, 255), 2, tipLength=0.3)
            cv2.putText(result_img, "Arrow = Adjustment movement", (legend_x + 25, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return result_img
            
        except Exception as e:
            print(f"Error creating after adjustment image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

