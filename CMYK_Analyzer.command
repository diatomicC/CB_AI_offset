#!/bin/bash

# CMYK Analyzer 실행 스크립트
# 이 파일을 더블클릭하면 프로그램이 실행됩니다.

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

# Python으로 GUI 실행
python3 run_gui.py

# 에러가 발생했을 경우 잠시 대기 (창이 바로 닫히지 않도록)
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 프로그램 실행 중 오류가 발생했습니다."
    echo "   터미널 창을 확인해주세요."
    echo ""
    read -p "계속하려면 Enter 키를 누르세요..."
fi

