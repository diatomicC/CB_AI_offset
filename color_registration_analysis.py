import cv2
import numpy as np
import glob
import os
import json

def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort four points in clockwise order (top-left, top-right, bottom-right, bottom-left)"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right  
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def extract_marker(image: np.ndarray) -> np.ndarray | None:
    """Extract marker area and perform perspective transform"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(255 - bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 5000: continue
        eps = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4,2)
            w = np.linalg.norm(pts[0] - pts[1])
            h = np.linalg.norm(pts[1] - pts[2])
            if abs(1 - w/h) < 0.3:
                cands.append((a, pts))
    
    if not cands:
        return None
    
    _, best = max(cands, key=lambda x: x[0])
    rect = order_points(best)
    size = int(max(
        np.linalg.norm(rect[0]-rect[1]),
        np.linalg.norm(rect[1]-rect[2]),
        np.linalg.norm(rect[2]-rect[3]),
        np.linalg.norm(rect[3]-rect[0])
    ))
    
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (size, size))

def detect_bottom_left(img: np.ndarray, hsv_range: tuple, min_area_ratio=0.5):
    """Detect bottom-left point of a color box (original function)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    
    areas = [cv2.contourArea(c) for c in cnts]
    mx = max(areas)
    big = [c for c,a in zip(cnts,areas) if a>=mx*min_area_ratio]
    
    for c in sorted(big, key=cv2.contourArea, reverse=True):
        hull = cv2.convexHull(c)
        eps = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True).reshape(-1,2)
        if len(approx)==4 and cv2.isContourConvex(approx):
            rect = order_points(approx)
            # Find bottom-left point (among points with large y, choose the one with smaller x)
            bottom = sorted(rect, key=lambda p:p[1], reverse=True)[:2]
            bl = tuple(sorted(bottom, key=lambda p:p[0])[0])
            return bl
    return None

def detect_square_corners(img: np.ndarray, hsv_range: tuple, min_area_ratio=0.5):
    """Detect all four corners of a color box (new function)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    
    areas = [cv2.contourArea(c) for c in cnts]
    mx = max(areas)
    big = [c for c,a in zip(cnts,areas) if a>=mx*min_area_ratio]
    
    for c in sorted(big, key=cv2.contourArea, reverse=True):
        hull = cv2.convexHull(c)
        eps = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True).reshape(-1,2)
        if len(approx)==4 and cv2.isContourConvex(approx):
            rect = order_points(approx)
            return rect  # [top-left, top-right, bottom-right, bottom-left]
    return None

def pixel_to_bottom_left_coord(x_px, y_px, img_height):
    """Convert pixel coordinates to coordinates based on bottom-left origin"""
    return x_px, img_height - y_px

def calculate_tilt_angle(bottom_left, bottom_right):
    """Calculate tilt angle (in degrees) using bottom-left and bottom-right vertices"""
    dx = bottom_right[0] - bottom_left[0]  # x difference
    dy = bottom_right[1] - bottom_left[1]  # y difference (in bottom-left origin coordinates)
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg, dx, dy

def calculate_horizontal_correction(bottom_left, bottom_right, square_width_px, mm_per_pixel):
    """Calculate vertical movement (in µm) needed for bottom-right point to be level with bottom-left"""
    dx = bottom_right[0] - bottom_left[0]
    dy = bottom_right[1] - bottom_left[1]
    
    correction_mm = -dy * mm_per_pixel  # negative means move down, positive means move up
    correction_um = correction_mm * 1000  # convert mm to µm
    
    return correction_um

def detect_special_color(img: np.ndarray, exclude_ranges: dict) -> tuple:
    """
    Dynamically detect a special color in the image.
    Find the largest color area excluding CMY color ranges.
    
    Args:
        img (np.ndarray): Input image (BGR)
        exclude_ranges (dict): HSV ranges to exclude (C, M, Y)
        
    Returns:
        tuple: (hsv_lower, hsv_upper) or None
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for excluded C, M, Y colors
    exclude_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color, (lower, upper) in exclude_ranges.items():
        if color in ['C', 'M', 'Y']:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            exclude_mask = cv2.bitwise_or(exclude_mask, mask)
    
    # Mask excluded areas
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_not(exclude_mask))
    
    # Test various color ranges to find the target
    color_ranges = [
        # Red tones
        ((0, 50, 50), (10, 255, 255)),
        ((170, 50, 50), (180, 255, 255)),
        # Orange tones
        ((10, 50, 50), (25, 255, 255)),
        # Green tones
        ((35, 50, 50), (85, 255, 255)),
        # Blue tones
        ((100, 50, 50), (130, 255, 255)),
        # Purple tones
        ((130, 50, 50), (170, 255, 255)),
        # Pink tones
        ((140, 30, 50), (170, 255, 255)),
        # Brown tones
        ((10, 100, 20), (20, 255, 200)),
        # Gray tones
        ((0, 0, 50), (180, 30, 200)),
    ]
    
    best_area = 0
    best_range = None
    
    for lower, upper in color_ranges:
        mask = cv2.inRange(masked_hsv, np.array(lower), np.array(upper))
        
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_area = max(cv2.contourArea(c) for c in contours)
            
            if max_area > best_area and max_area > 1000:
                best_area = max_area
                best_range = (lower, upper)
    
    return best_range


def main():
    # HSV color ranges (only CMY defined here, S will be detected dynamically)
    HSV = {
        'C': ((90,80,80),(130,255,255)),   # Cyan
        'M': ((130,50,70),(170,255,255)),  # Magenta
        'Y': ((20,80,80),(40,255,255)),    # Yellow
    }
    
    # Select analysis mode
    print("\n📋 Select analysis mode:")
    print("1. Standard analysis (Color registration)")
    print("2. Tilt analysis (CMYK square tilt)")
    print("3. Run both")
    
    # Set default for testing
    mode = '3'  # Run both
    print(f"Test mode: {mode} (Run both)")
    
    # Create directories for saving results
    os.makedirs('./registration_analysis', exist_ok=True)
    os.makedirs('./registration_analysis/debug', exist_ok=True)

    for path in glob.glob('./output/extracted_*.png'):
        print(f"\n🔍 Analyzing: {path}")
        
        orig = cv2.imread(path)
        if orig is None:
            print(f"❌ Unable to read image: {path}")
            continue
            
        cropped = extract_marker(orig)
        if cropped is None:
            print(f"❌ Marker not found: {path}")
            continue

        h_px, w_px = cropped.shape[:2]
        print(f"📐 Image size: {w_px} x {h_px} pixels")
        
        # Conversion ratio for 5mm x 5mm
        mm_per_pixel_x = 5.0 / w_px
        mm_per_pixel_y = 5.0 / h_px
        
        # ===========================================
        # 1. Standard analysis (Color registration)
        # ===========================================
        if mode in ['1', '3']:
            print(f"\n📍 Color registration analysis:")
            
            # Detect special color
            print("🔍 Detecting special color...")
            special_color_range = detect_special_color(cropped, HSV)
            
            if special_color_range is None:
                print("❌ Unable to detect special color.")
                continue
            
            # Add special color to HSV ranges
            HSV['S'] = special_color_range
            print(f"✅ Special color detected: HSV range {special_color_range}")
            
            # Target coordinates (based on bottom-left origin (0,0))
            target_coords = {
                'S': (w_px/10, h_px - h_px*6/10),    # Special color in K position
                'C': (w_px*6/10, h_px - h_px*6/10),  # (length*6/10, height*4/10) from bottom
                'M': (w_px/10, h_px - h_px/10),      # (length/10, height*9/10) from bottom
                'Y': (w_px*6/10, h_px - h_px/10)     # (length*6/10, height*9/10) from bottom
            }
            
            results_reg = {}
            debug_reg = cropped.copy()
            
            # Draw red X on target points
            for color, (tx_px, ty_px) in target_coords.items():
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                tx_cv = int(tx_bl)
                ty_cv = int(h_px - ty_bl)  # Convert back to OpenCV coordinates
                
                cv2.line(debug_reg, (tx_cv-10, ty_cv-10), (tx_cv+10, ty_cv+10), (0,0,255), 2)
                cv2.line(debug_reg, (tx_cv-10, ty_cv+10), (tx_cv+10, ty_cv-10), (0,0,255), 2)
                cv2.putText(debug_reg, f"T{color}", (tx_cv+12, ty_cv), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Detect P coordinates (actual bottom-left of each color box)
            for color, hsv_range in HSV.items():
                bl = detect_bottom_left(cropped, hsv_range)
                if bl is None:
                    print(f"❌ {color} color box not found")
                    results_reg[color] = None
                    continue
                
                px_px, py_px = bl  # Pixel coordinates
                px_bl, py_bl = pixel_to_bottom_left_coord(px_px, py_px, h_px)  # Bottom-left origin
                
                # Target coordinate (T)
                tx_px, ty_px = target_coords[color]
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                
                # Calculate movement (P -> T)
                dx_px = tx_bl - px_bl  # right +, left -
                dy_px = ty_bl - py_bl  # up +, down -
                
                # Convert to mm
                px_mm = px_bl * mm_per_pixel_x
                py_mm = py_bl * mm_per_pixel_y
                tx_mm = tx_bl * mm_per_pixel_x  
                ty_mm = ty_bl * mm_per_pixel_y
                dx_mm = dx_px * mm_per_pixel_x
                dy_mm = dy_px * mm_per_pixel_y
                
                results_reg[color] = {
                    'P_coord_mm': (round(px_mm, 2), round(py_mm, 2)),
                    'T_coord_mm': (round(tx_mm, 2), round(ty_mm, 2)),
                    'movement_mm': (round(dx_mm, 2), round(dy_mm, 2))
                }
                
                # Debug: mark P coordinate with green dot
                px_int, py_int = int(px_px), int(py_px)
                cv2.circle(debug_reg, (px_int, py_int), 8, (0,255,0), -1)
                cv2.putText(debug_reg, f"P{color}({px_mm:.2f},{py_mm:.2f})", 
                           (px_int+15, py_int-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(debug_reg, f"Move({dx_mm:.2f},{dy_mm:.2f})", 
                           (px_int+15, py_int+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                print(f"✅ {color}: P({px_mm:.2f}, {py_mm:.2f})mm -> T({tx_mm:.2f}, {ty_mm:.2f})mm, movement({dx_mm:.2f}, {dy_mm:.2f})mm")

            # Save registration results
            name = os.path.basename(path)
            result_path_reg = f"./registration_analysis/{name}.json"
            debug_path_reg = f"./registration_analysis/debug/dbg_{name}"
            
            with open(result_path_reg, "w") as f:
                json.dump(results_reg, f, indent=2, ensure_ascii=False)
            
            cv2.imwrite(debug_path_reg, debug_reg)
            print(f"💾 Registration result: {result_path_reg}")
            print(f"🖼️  Registration debug: {debug_path_reg}")
        
        # ===========================================
        # 2. Tilt analysis
        # ===========================================
        if mode in ['2', '3']:
            # Get total print width from user
            if mode == '2':
                print("\n📏 Enter total print width (in mm):")
                print("Using default value 210mm for testing (A4 width)")
            total_width_mm = 210.0  # Default A4 width
            if mode == '2':
                print(f"Using width: {total_width_mm}mm")
            
            print(f"\n🔍 Tilt analysis of each CMYK square:")
            
            results_tilt = {}
            debug_tilt = cropped.copy()
            
            # Detect square corners for each color and analyze tilt
            for color, hsv_range in HSV.items():
                corners = detect_square_corners(cropped, hsv_range)
                if corners is None:
                    print(f"❌ {color} color box not found")
                    results_tilt[color] = None
                    continue
                
                # corners: [top-left, top-right, bottom-right, bottom-left]
                tl, tr, br, bl = corners
                
                # Convert pixel coordinates to bottom-left origin
                bl_coord = pixel_to_bottom_left_coord(bl[0], bl[1], h_px)
                br_coord = pixel_to_bottom_left_coord(br[0], br[1], h_px)
                
                # Calculate tilt angle
                angle_deg, dx_px, dy_px = calculate_tilt_angle(bl_coord, br_coord)
                
                # Square width (pixels)
                square_width_px = np.linalg.norm(br - bl)
                
                # Horizontal correction (µm)
                correction_um = calculate_horizontal_correction(bl_coord, br_coord, square_width_px, mm_per_pixel_x)
                
                # Scale to actual print size: 5mm : total_width_mm = correction_um : final_correction_um
                final_correction_um = (correction_um * total_width_mm) / 5.0
                
                results_tilt[color] = {
                    'bottom_left_px': [float(bl[0]), float(bl[1])],
                    'bottom_right_px': [float(br[0]), float(br[1])],
                    'bottom_left_coord': [float(bl_coord[0]), float(bl_coord[1])],
                    'bottom_right_coord': [float(br_coord[0]), float(br_coord[1])],
                    'tilt_angle_deg': round(float(angle_deg), 6),
                    'dx_px': round(float(dx_px), 2),
                    'dy_px': round(float(dy_px), 2),
                    'correction_5mm_um': round(float(correction_um), 2),
                    'correction_actual_um': round(float(final_correction_um), 2)
                }
                
                # Debug visualization
                # Bottom-left vertex (reference) - red circle
                cv2.circle(debug_tilt, (int(bl[0]), int(bl[1])), 8, (0,0,255), -1)
                cv2.putText(debug_tilt, f"{color}_BL", (int(bl[0])+10, int(bl[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
                # Bottom-right vertex - blue circle
                cv2.circle(debug_tilt, (int(br[0]), int(br[1])), 8, (255,0,0), -1)
                cv2.putText(debug_tilt, f"{color}_BR", (int(br[0])+10, int(br[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
                # Base line
                cv2.line(debug_tilt, (int(bl[0]), int(bl[1])), (int(br[0]), int(br[1])), (0,255,0), 2)
                
                # Horizontal reference line
                cv2.line(debug_tilt, (int(bl[0]), int(bl[1])), (int(bl[0] + square_width_px), int(bl[1])), (255,255,0), 1)
                
                # Display results
                text_y = int(bl[1]) + 30
                cv2.putText(debug_tilt, f"Angle: {angle_deg:.4f}°", (int(bl[0]), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(debug_tilt, f"Corr: {final_correction_um:.1f}μm", (int(bl[0]), text_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                print(f"✅ {color} square:")
                print(f"  • Bottom-left: ({bl_coord[0]:.1f}, {bl_coord[1]:.1f})")
                print(f"  • Bottom-right: ({br_coord[0]:.1f}, {br_coord[1]:.1f})")
                print(f"  • Tilt angle: {angle_deg:.6f}°")
                print(f"  • dx: {dx_px:.2f}px, dy: {dy_px:.2f}px")
                print(f"  • Correction (5mm ref): {correction_um:.2f}µm")
                print(f"  • Correction ({total_width_mm}mm ref): {final_correction_um:.2f}µm")
                if final_correction_um > 0:
                    print(f"  • Correction direction: Move bottom-right vertex up by {abs(final_correction_um):.1f}µm")
                else:
                    print(f"  • Correction direction: Move bottom-right vertex down by {abs(final_correction_um):.1f}µm")

            # Save tilt analysis results
            name = os.path.basename(path)
            result_path_tilt = f"./registration_analysis/tilt_{name}.json"
            debug_path_tilt = f"./registration_analysis/debug/tilt_dbg_{name}"
            
            results_tilt['metadata'] = {
                'total_width_mm': total_width_mm,
                'image_size_px': [w_px, h_px],
                'mm_per_pixel': [mm_per_pixel_x, mm_per_pixel_y]
            }
            
            with open(result_path_tilt, "w") as f:
                json.dump(results_tilt, f, indent=2, ensure_ascii=False)
            
            cv2.imwrite(debug_path_tilt, debug_tilt)
            print(f"💾 Tilt analysis result: {result_path_tilt}")
            print(f"🖼️  Tilt analysis debug: {debug_path_tilt}")

if __name__ == "__main__":
    main()
