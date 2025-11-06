#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color registration analysis helper functions
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def extract_marker(image: np.ndarray) -> Optional[np.ndarray]:
    """Extract the largest square marker region with perspective transform."""
    if image is None or image.size == 0:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(255 - bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        eps = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype("float32")
            w = np.linalg.norm(pts[0] - pts[1])
            h = np.linalg.norm(pts[1] - pts[2])
            if h == 0:
                continue
            if abs(1.0 - (w / h)) < 0.3:
                candidates.append((area, pts))
    
    if not candidates:
        return None
    
    _, best = max(candidates, key=lambda x: x[0])
    rect = order_points(best)
    size = int(max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[1] - rect[2]),
        np.linalg.norm(rect[2] - rect[3]),
        np.linalg.norm(rect[3] - rect[0])
    ))
    if size <= 0:
        return None
    
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (size, size))


def pixel_to_bottom_left_coord(x_px: float, y_px: float, image_height_px: int) -> Tuple[float, float]:
    """Convert pixel coordinates (top-left origin) to bottom-left origin."""
    return float(x_px), float(image_height_px - y_px)


def detect_square_corners(img: np.ndarray, 
                          hsv_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]], 
                          min_area_ratio: float = 0.5) -> Optional[np.ndarray]:
    """Detect a colored square and return its 4 corners ordered."""
    if img is None or img.size == 0:
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = hsv_range
    lower_np = np.array(lower, dtype=np.uint8)
    upper_np = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_np, upper_np)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    img_area = img.shape[0] * img.shape[1]
    best = None
    best_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * 0.01 * img_area:
            continue
        eps = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best = approx.reshape(4, 2).astype("float32")
    
    if best is None:
        return None
    
    return order_points(best)


def detect_bottom_left(img: np.ndarray,
                       hsv_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
                       min_area_ratio: float = 0.5) -> Optional[Tuple[int, int]]:
    """Detect bottom-left corner of the colored square."""
    rect = detect_square_corners(img, hsv_range, min_area_ratio=min_area_ratio)
    if rect is None:
        return None
    bl = rect[3]
    return int(bl[0]), int(bl[1])
