#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMYS Analyzer GUI 테스트 스크립트
GUI 없이 분석 기능이 제대로 작동하는지 테스트합니다.
"""

import sys
import os
import json
import cv2
import numpy as np
from datetime import datetime

# 기존 분석 모듈 import
from color_registration_analysis import (
    extract_marker, detect_bottom_left, detect_square_corners,
    pixel_to_bottom_left_coord, calculate_tilt_angle, 
    calculate_horizontal_correction, order_points, detect_special_color
)

def test_analysis(image_path, print_width_mm=210.0):
    """분석 기능 테스트"""
    print(f"🔍 테스트 이미지: {image_path}")
    print(f"📏 프린트물 가로길이: {print_width_mm}mm")
    print("-" * 50)
    
    try:
        print("1️⃣ 이미지 로딩 중...")
        orig = cv2.imread(image_path)
        if orig is None:
            print("❌ 이미지를 읽을 수 없습니다.")
            return False
            
        print("2️⃣ 마커 영역 추출 중...")
        cropped = extract_marker(orig)
        if cropped is None:
            print("❌ 마커를 찾을 수 없습니다.")
            return False
            
        h_px, w_px = cropped.shape[:2]
        mm_per_pixel_x = 5.0 / w_px
        mm_per_pixel_y = 5.0 / h_px
        
        print(f"✅ 마커 추출 완료: {w_px} x {h_px} pixels")
        print(f"📐 픽셀당 mm: {mm_per_pixel_x:.6f} x {mm_per_pixel_y:.6f}")
        
        # HSV 색상 범위 (CMY만 정의, S는 동적 감지)
        HSV = {
            'C': ((90,80,80),(130,255,255)),   # 청록색 (Cyan)
            'M': ((130,50,70),(170,255,255)),  # 자홍색 (Magenta)
            'Y': ((20,80,80),(40,255,255)),    # 노란색 (Yellow)
        }
        
        # Special color 감지
        print("🔍 특별한 색상 감지 중...")
        special_color_range = detect_special_color(cropped, HSV)
        
        if special_color_range is None:
            print("❌ 특별한 색상을 감지할 수 없습니다.")
            return False
        
        # HSV에 특별한 색상 추가
        HSV['S'] = special_color_range
        print(f"✅ 특별한 색상 감지됨: HSV 범위 {special_color_range}")
        
        # 목표 좌표 (왼쪽 아래 기준)
        target_coords = {
            'S': (w_px/10, h_px - h_px*6/10),  # Special color in K position
            'C': (w_px*6/10, h_px - h_px*6/10),
            'M': (w_px/10, h_px - h_px/10),
            'Y': (w_px*6/10, h_px - h_px/10)
        }
        
        print("\n3️⃣ 컬러 레지스트레이션 분석 중...")
        results_reg = {}
        
        for color, hsv_range in HSV.items():
            print(f"  🎨 {color} 색상 분석 중...")
            bl = detect_bottom_left(cropped, hsv_range)
            if bl is None:
                print(f"    ❌ {color} 색상 박스를 찾을 수 없습니다")
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
            
            print(f"    ✅ P({px_mm:.3f}, {py_mm:.3f})mm -> T({tx_mm:.3f}, {ty_mm:.3f})mm")
            print(f"    ➡️  이동량: ({dx_mm:+.3f}, {dy_mm:+.3f})mm")
        
        print("\n4️⃣ 기울기 분석 중...")
        results_tilt = {}
        
        for color, hsv_range in HSV.items():
            print(f"  🎨 {color} 색상 기울기 분석 중...")
            corners = detect_square_corners(cropped, hsv_range)
            if corners is None:
                print(f"    ❌ {color} 색상 박스를 찾을 수 없습니다")
                results_tilt[color] = None
                continue
            
            tl, tr, br, bl = corners
            
            bl_coord = pixel_to_bottom_left_coord(bl[0], bl[1], h_px)
            br_coord = pixel_to_bottom_left_coord(br[0], br[1], h_px)
            
            angle_deg, dx_px, dy_px = calculate_tilt_angle(bl_coord, br_coord)
            square_width_px = np.linalg.norm(br - bl)
            correction_um = calculate_horizontal_correction(bl_coord, br_coord, square_width_px, mm_per_pixel_x)
            final_correction_um = (correction_um * print_width_mm) / 5.0
            
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
            
            print(f"    ✅ 기울기 각도: {angle_deg:+.6f}°")
            print(f"    🔧 보정값: {final_correction_um:+.3f}μm")
            
            if abs(final_correction_um) > 0.1:
                direction = "위로" if final_correction_um > 0 else "아래로"
                print(f"    ➡️  오른쪽 아래 꼭지점을 {abs(final_correction_um):.1f}μm {direction} 이동 필요")
            else:
                print(f"    ✅ 기울기 보정 불필요 (오차 < 0.1μm)")
        
        print("\n5️⃣ 결과 요약:")
        print("=" * 50)
        
        for color in ['C', 'M', 'Y', 'S']:
            print(f"\n🎨 {color} 색상:")
            
            # 레지스트레이션 결과
            if results_reg.get(color):
                reg = results_reg[color]
                dx, dy = reg['movement_mm']
                print(f"  📍 레지스트레이션: ({dx:+.3f}, {dy:+.3f}) mm")
            else:
                print(f"  ❌ 레지스트레이션: 검출 실패")
                
            # 기울기 결과
            if results_tilt.get(color):
                tilt = results_tilt[color]
                angle = tilt['tilt_angle_deg']
                corr = tilt['correction_actual_um']
                print(f"  📐 기울기: {angle:+.6f}° (보정: {corr:+.1f} μm)")
            else:
                print(f"  ❌ 기울기: 검출 실패")
        
        print("\n✅ 분석 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🎯 CMYS Registration & Tilt Analyzer 테스트")
    print("=" * 60)
    
    # 테스트할 이미지 파일들
    test_images = [
        "output/extracted_left_1.png",
        "output/extracted_right_1.png",
        "output/extracted_20250718_145610755.png"
    ]
    
    success_count = 0
    total_count = len(test_images)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            if test_analysis(image_path):
                success_count += 1
        else:
            print(f"\n❌ 파일을 찾을 수 없습니다: {image_path}")
    
    print(f"\n{'='*60}")
    print(f"📊 테스트 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 테스트가 성공했습니다!")
        print("✅ GUI 애플리케이션을 실행할 준비가 되었습니다.")
        print("💡 python cmyk_analyzer_gui.py 명령으로 GUI를 실행하세요.")
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
        print("🔧 문제를 해결한 후 다시 시도하세요.")

if __name__ == "__main__":
    main() 