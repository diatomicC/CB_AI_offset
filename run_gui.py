#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMYK Registration & Tilt Analyzer GUI 실행 스크립트
"""

import sys
import os
import subprocess

def check_dependencies():
    """필요한 패키지들이 설치되어 있는지 확인"""
    required_packages = [
        'PySide6',
        'opencv-python',
        'numpy',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'Pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 다음 명령으로 설치하세요:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """메인 함수"""
    print("🎯 CMYK Registration & Tilt Analyzer GUI")
    print("=" * 50)
    
    # 현재 디렉토리 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gui_file = os.path.join(current_dir, "cmyk_analyzer_gui.py")
    
    if not os.path.exists(gui_file):
        print(f"❌ GUI 파일을 찾을 수 없습니다: {gui_file}")
        return
    
    # 의존성 확인
    print("🔍 의존성 확인 중...")
    if not check_dependencies():
        return
    
    print("✅ 모든 의존성이 설치되어 있습니다.")
    print("🚀 GUI 애플리케이션을 시작합니다...")
    print("-" * 50)
    
    try:
        # GUI 실행
        subprocess.run([sys.executable, gui_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ GUI 실행 중 오류가 발생했습니다: {e}")
    except KeyboardInterrupt:
        print("\n👋 GUI가 종료되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 