import cv2
import numpy as np
import glob
import os
import json

def order_points(pts: np.ndarray) -> np.ndarray:
    """4개 점을 시계방향으로 정렬 (top-left, top-right, bottom-right, bottom-left)"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right  
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def extract_marker(image: np.ndarray) -> np.ndarray | None:
    """마커 영역 추출 및 원근 변환"""
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
    """컬러 박스의 왼쪽 아래 점 검출 (기존 함수)"""
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
            # 왼쪽 아래 점 찾기 (y가 큰 점들 중에서 x가 작은 점)
            bottom = sorted(rect, key=lambda p:p[1], reverse=True)[:2]
            bl = tuple(sorted(bottom, key=lambda p:p[0])[0])
            return bl
    return None

def detect_square_corners(img: np.ndarray, hsv_range: tuple, min_area_ratio=0.5):
    """컬러 박스의 네 꼭지점 검출 (새 함수)"""
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
    """픽셀 좌표를 왼쪽 아래 기준 좌표계로 변환"""
    return x_px, img_height - y_px

def calculate_tilt_angle(bottom_left, bottom_right):
    """왼쪽 아래와 오른쪽 아래 꼭지점으로 기울기 각도 계산 (도 단위)"""
    dx = bottom_right[0] - bottom_left[0]  # x 차이
    dy = bottom_right[1] - bottom_left[1]  # y 차이 (왼쪽 아래 기준 좌표계)
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg, dx, dy

def calculate_horizontal_correction(bottom_left, bottom_right, square_width_px, mm_per_pixel):
    """수평 보정을 위해 오른쪽 아래 점이 움직여야 할 거리 계산 (µm 단위)"""
    dx = bottom_right[0] - bottom_left[0]
    dy = bottom_right[1] - bottom_left[1]
    
    # 오른쪽 아래 점을 왼쪽 아래 점과 같은 y 좌표로 만들기 위한 수직 이동량
    correction_mm = -dy * mm_per_pixel  # 음수면 아래로, 양수면 위로
    correction_um = correction_mm * 1000  # mm를 µm로 변환
    
    return correction_um

def detect_special_color(img: np.ndarray, exclude_ranges: dict) -> tuple:
    """
    이미지에서 특별한 색상을 동적으로 감지합니다.
    CMY 색상 범위를 제외한 영역에서 가장 큰 색상 영역을 찾습니다.
    
    Args:
        img (np.ndarray): 입력 이미지 (BGR)
        exclude_ranges (dict): 제외할 색상 범위들 (C, M, Y)
        
    Returns:
        tuple: (hsv_lower, hsv_upper) 또는 None
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # C, M, Y 색상 마스크 생성
    exclude_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color, (lower, upper) in exclude_ranges.items():
        if color in ['C', 'M', 'Y']:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            exclude_mask = cv2.bitwise_or(exclude_mask, mask)
    
    # 제외 영역을 마스킹한 이미지
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_not(exclude_mask))
    
    # 색상별로 영역을 찾기 위해 다양한 색상 범위를 테스트
    color_ranges = [
        # 빨간색 계열
        ((0, 50, 50), (10, 255, 255)),
        ((170, 50, 50), (180, 255, 255)),
        # 주황색 계열
        ((10, 50, 50), (25, 255, 255)),
        # 초록색 계열
        ((35, 50, 50), (85, 255, 255)),
        # 파란색 계열
        ((100, 50, 50), (130, 255, 255)),
        # 보라색 계열
        ((130, 50, 50), (170, 255, 255)),
        # 분홍색 계열
        ((140, 30, 50), (170, 255, 255)),
        # 갈색 계열
        ((10, 100, 20), (20, 255, 200)),
        # 회색 계열
        ((0, 0, 50), (180, 30, 200)),
    ]
    
    best_area = 0
    best_range = None
    
    for lower, upper in color_ranges:
        mask = cv2.inRange(masked_hsv, np.array(lower), np.array(upper))
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 컨투어의 면적 계산
            max_area = max(cv2.contourArea(c) for c in contours)
            
            if max_area > best_area and max_area > 1000:  # 최소 면적 조건
                best_area = max_area
                best_range = (lower, upper)
    
    return best_range

def main():
    # HSV 색상 범위 (CMY만 정의, S는 동적 감지)
    HSV = {
        'C': ((90,80,80),(130,255,255)),   # 청록색 (Cyan)
        'M': ((130,50,70),(170,255,255)),  # 자홍색 (Magenta) 
        'Y': ((20,80,80),(40,255,255)),    # 노란색 (Yellow)
    }
    
    # 분석 모드 선택
    print("\n📋 분석 모드를 선택하세요:")
    print("1. 기존 분석 (컬러 레지스트레이션)")
    print("2. 기울기 분석 (CMYK 정사각형 기울기)")
    print("3. 둘 다 실행")
    
    # 테스트를 위해 기본값 설정
    mode = '3'  # 둘 다 실행
    print(f"테스트 모드: {mode} (둘 다 실행)")
    
    # 결과 저장용 디렉토리 생성
    os.makedirs('./registration_analysis', exist_ok=True)
    os.makedirs('./registration_analysis/debug', exist_ok=True)

    for path in glob.glob('./output/extracted_*.png'):
        print(f"\n🔍 분석 중: {path}")
        
        orig = cv2.imread(path)
        if orig is None:
            print(f"❌ 이미지를 읽을 수 없습니다: {path}")
            continue
            
        cropped = extract_marker(orig)
        if cropped is None:
            print(f"❌ 마커를 찾을 수 없습니다: {path}")
            continue

        h_px, w_px = cropped.shape[:2]
        print(f"📐 이미지 크기: {w_px} x {h_px} pixels")
        
        # 5mm x 5mm로 변환하는 비율
        mm_per_pixel_x = 5.0 / w_px
        mm_per_pixel_y = 5.0 / h_px
        
        # ===========================================
        # 1. 기존 분석 (컬러 레지스트레이션)
        # ===========================================
        if mode in ['1', '3']:
            print(f"\n📍 컬러 레지스트레이션 분석:")
            
            # Special color 감지
            print("🔍 특별한 색상 감지 중...")
            special_color_range = detect_special_color(cropped, HSV)
            
            if special_color_range is None:
                print("❌ 특별한 색상을 감지할 수 없습니다.")
                continue
            
            # HSV에 특별한 색상 추가
            HSV['S'] = special_color_range
            print(f"✅ 특별한 색상 감지됨: HSV 범위 {special_color_range}")
            
            # T 좌표 (목표 기준점들) - 왼쪽 아래 (0,0) 기준으로 픽셀 좌표 계산
            target_coords = {
                'S': (w_px/10, h_px - h_px*6/10),    # Special color in K position
                'C': (w_px*6/10, h_px - h_px*6/10),  # (length*6/10, height*4/10) - 아래서부터  
                'M': (w_px/10, h_px - h_px/10),      # (length/10, height*9/10) - 아래서부터
                'Y': (w_px*6/10, h_px - h_px/10)     # (length*6/10, height*9/10) - 아래서부터
            }
            
            results_reg = {}
            debug_reg = cropped.copy()
            
            # 목표 기준점들을 빨간색 X로 표시
            for color, (tx_px, ty_px) in target_coords.items():
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                tx_cv = int(tx_bl)
                ty_cv = int(h_px - ty_bl)  # OpenCV 좌표계로 다시 변환
                
                cv2.line(debug_reg, (tx_cv-10, ty_cv-10), (tx_cv+10, ty_cv+10), (0,0,255), 2)
                cv2.line(debug_reg, (tx_cv-10, ty_cv+10), (tx_cv+10, ty_cv-10), (0,0,255), 2)
                cv2.putText(debug_reg, f"T{color}", (tx_cv+12, ty_cv), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # 각 색상의 P 좌표 (실제 컬러박스 왼쪽 아래) 검출
            for color, hsv_range in HSV.items():
                bl = detect_bottom_left(cropped, hsv_range)
                if bl is None:
                    print(f"❌ {color} 색상 박스를 찾을 수 없습니다")
                    results_reg[color] = None
                    continue
                
                px_px, py_px = bl  # 픽셀 좌표계에서의 위치
                px_bl, py_bl = pixel_to_bottom_left_coord(px_px, py_px, h_px)  # 왼쪽 아래 기준
                
                # T 좌표 (목표점)
                tx_px, ty_px = target_coords[color]
                tx_bl, ty_bl = pixel_to_bottom_left_coord(tx_px, ty_px, h_px)
                
                # 이동량 계산 (P -> T)
                dx_px = tx_bl - px_bl  # 오른쪽 +, 왼쪽 -
                dy_px = ty_bl - py_bl  # 위쪽 +, 아래쪽 -
                
                # mm 단위로 변환
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
                
                # 디버깅: P 좌표에 초록색 점 표시
                px_int, py_int = int(px_px), int(py_px)
                cv2.circle(debug_reg, (px_int, py_int), 8, (0,255,0), -1)
                cv2.putText(debug_reg, f"P{color}({px_mm:.2f},{py_mm:.2f})", 
                           (px_int+15, py_int-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(debug_reg, f"Move({dx_mm:.2f},{dy_mm:.2f})", 
                           (px_int+15, py_int+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                print(f"✅ {color}: P({px_mm:.3f}, {py_mm:.3f})mm -> T({tx_mm:.3f}, {ty_mm:.3f})mm, 이동량({dx_mm:.3f}, {dy_mm:.3f})mm")

            # 기존 분석 결과 저장
            name = os.path.basename(path)
            result_path_reg = f"./registration_analysis/{name}.json"
            debug_path_reg = f"./registration_analysis/debug/dbg_{name}"
            
            with open(result_path_reg, "w") as f:
                json.dump(results_reg, f, indent=2, ensure_ascii=False)
            
            cv2.imwrite(debug_path_reg, debug_reg)
            print(f"💾 레지스트레이션 결과: {result_path_reg}")
            print(f"🖼️  레지스트레이션 디버그: {debug_path_reg}")
        
        # ===========================================
        # 2. 기울기 분석
        # ===========================================
        if mode in ['2', '3']:
            # 사용자로부터 전체 프린트물 가로길이 입력받기
            if mode == '2':
                print("\n📏 전체 프린트물의 가로길이를 입력해주세요 (mm 단위):")
                print("테스트를 위해 기본값 210mm를 사용합니다 (A4 가로)")
            total_width_mm = 210.0  # A4 가로 기본값
            if mode == '2':
                print(f"사용된 가로길이: {total_width_mm}mm")
            
            print(f"\n🔍 각 CMYK 정사각형의 기울기 분석:")
            
            results_tilt = {}
            debug_tilt = cropped.copy()
            
            # 각 색상의 정사각형 꼭지점 검출 및 기울기 분석
            for color, hsv_range in HSV.items():
                corners = detect_square_corners(cropped, hsv_range)
                if corners is None:
                    print(f"❌ {color} 색상 박스를 찾을 수 없습니다")
                    results_tilt[color] = None
                    continue
                
                # corners: [top-left, top-right, bottom-right, bottom-left]
                tl, tr, br, bl = corners
                
                # 픽셀 좌표를 왼쪽 아래 기준 좌표계로 변환
                bl_coord = pixel_to_bottom_left_coord(bl[0], bl[1], h_px)
                br_coord = pixel_to_bottom_left_coord(br[0], br[1], h_px)
                
                # 기울기 각도 계산
                angle_deg, dx_px, dy_px = calculate_tilt_angle(bl_coord, br_coord)
                
                # 정사각형 너비 (픽셀)
                square_width_px = np.linalg.norm(br - bl)
                
                # 수평 보정값 계산 (µm)
                correction_um = calculate_horizontal_correction(bl_coord, br_coord, square_width_px, mm_per_pixel_x)
                
                # 실제 프린트물 크기로 비례 계산: 5mm : total_width_mm = correction_um : final_correction_um
                final_correction_um = (correction_um * total_width_mm) / 5.0
                
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
                
                # 디버깅 시각화
                # 왼쪽 아래 꼭지점 (기준점) - 빨간색 원
                cv2.circle(debug_tilt, (int(bl[0]), int(bl[1])), 8, (0,0,255), -1)
                cv2.putText(debug_tilt, f"{color}_BL", (int(bl[0])+10, int(bl[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
                # 오른쪽 아래 꼭지점 - 파란색 원
                cv2.circle(debug_tilt, (int(br[0]), int(br[1])), 8, (255,0,0), -1)
                cv2.putText(debug_tilt, f"{color}_BR", (int(br[0])+10, int(br[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
                # 밑변 선 그리기
                cv2.line(debug_tilt, (int(bl[0]), int(bl[1])), (int(br[0]), int(br[1])), (0,255,0), 2)
                
                # 수평선 그리기 (기준)
                cv2.line(debug_tilt, (int(bl[0]), int(bl[1])), (int(bl[0] + square_width_px), int(bl[1])), (255,255,0), 1)
                
                # 결과 텍스트 표시
                text_y = int(bl[1]) + 30
                cv2.putText(debug_tilt, f"Angle: {angle_deg:.4f}°", (int(bl[0]), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(debug_tilt, f"Corr: {final_correction_um:.1f}μm", (int(bl[0]), text_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                print(f"✅ {color} 정사각형:")
                print(f"  • 왼쪽 아래: ({bl_coord[0]:.1f}, {bl_coord[1]:.1f})")
                print(f"  • 오른쪽 아래: ({br_coord[0]:.1f}, {br_coord[1]:.1f})")
                print(f"  • 기울기 각도: {angle_deg:.6f}°")
                print(f"  • dx: {dx_px:.3f}px, dy: {dy_px:.3f}px")
                print(f"  • 5mm 기준 보정값: {correction_um:.3f}µm")
                print(f"  • {total_width_mm}mm 기준 보정값: {final_correction_um:.3f}µm")
                if final_correction_um > 0:
                    print(f"  • 보정 방향: 오른쪽 아래 꼭지점을 {abs(final_correction_um):.1f}µm 위로 이동")
                else:
                    print(f"  • 보정 방향: 오른쪽 아래 꼭지점을 {abs(final_correction_um):.1f}µm 아래로 이동")

            # 기울기 분석 결과 저장
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
            print(f"💾 기울기 분석 결과: {result_path_tilt}")
            print(f"🖼️  기울기 분석 디버그: {debug_path_tilt}")

if __name__ == "__main__":
    main()