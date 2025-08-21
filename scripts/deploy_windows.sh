#!/bin/bash

# CMYK Analyzer Windows 배포 스크립트

echo "🚀 CMYK Analyzer Windows 배포 시작"
echo "=================================================="

# 빌드 확인
if [ ! -d "dist" ] || [ ! -f "dist/CMYK_Analyzer.exe" ]; then
    echo "❌ Windows 실행파일을 찾을 수 없습니다."
    echo "   먼저 Windows 빌드를 실행하세요: python build_windows.py"
    exit 1
fi

# 버전 정보 입력
read -p "📝 배포 버전을 입력하세요 (예: 1.0.0): " VERSION
read -p "📝 배포 노트를 입력하세요: " RELEASE_NOTES

# 배포 디렉토리 생성
DEPLOY_DIR="deploy_windows_${VERSION}"
mkdir -p "$DEPLOY_DIR"

echo "📁 배포 디렉토리 생성: $DEPLOY_DIR"

# 1. 실행파일 복사
echo "📦 실행파일 복사 중..."
cp "dist/CMYK_Analyzer.exe" "$DEPLOY_DIR/CMYK_Analyzer_${VERSION}.exe"

# 2. 아이콘 파일 복사
if [ -f "app_icon.ico" ]; then
    echo "🎨 아이콘 파일 복사 중..."
    cp "app_icon.ico" "$DEPLOY_DIR/"
fi

# 3. ZIP 파일 생성
echo "🗜️ ZIP 파일 생성 중..."
cd "$DEPLOY_DIR"
zip -r "CMYK_Analyzer_Windows_${VERSION}.zip" *
cd ..

# 4. 설치 프로그램 복사 (있는 경우)
if [ -f "CMYK_Analyzer_Setup.exe" ]; then
    echo "📦 설치 프로그램 복사 중..."
    cp "CMYK_Analyzer_Setup.exe" "$DEPLOY_DIR/"
fi

# 5. README 파일 생성
echo "📖 README 파일 생성 중..."
cat > "$DEPLOY_DIR/README.txt" << EOF
CMYK Registration & Tilt Analyzer ${VERSION}
==================================================

🪟 Windows 실행파일 배포

📦 포함된 파일:
- CMYK_Analyzer_${VERSION}.exe: Windows 실행파일
- app_icon.ico: 애플리케이션 아이콘
- CMYK_Analyzer_Windows_${VERSION}.zip: 압축파일
${if [ -f "CMYK_Analyzer_Setup.exe" ]; then echo "- CMYK_Analyzer_Setup.exe: 설치 프로그램"; fi}

🚀 설치 방법:

방법 1: 직접 실행
- CMYK_Analyzer_${VERSION}.exe를 더블클릭하여 실행

방법 2: 설치 프로그램 사용 (권장)
- CMYK_Analyzer_Setup.exe를 실행하여 설치
- 시작 메뉴와 바탕화면에 바로가기 자동 생성

방법 3: ZIP 파일 압축 해제
- CMYK_Analyzer_Windows_${VERSION}.zip 압축 해제
- 원하는 위치에 배치하여 사용

📋 시스템 요구사항:
- Windows 10 이상 (64비트)
- .NET Framework 4.7.2 이상
- 최소 4GB RAM
- 500MB 이상의 디스크 공간

🔧 문제 해결:
- 실행이 안 되는 경우: 바이러스 백신 프로그램에서 예외 처리
- 권한 문제: 관리자 권한으로 실행
- DLL 오류: Visual C++ Redistributable 설치

📝 배포 노트:
${RELEASE_NOTES}

📞 지원:
- GitHub Issues: [저장소 URL]
- 이메일: [이메일 주소]

© 2025 CMYK Analyzer Team
EOF

# 6. 체크섬 생성
echo "🔒 체크섬 생성 중..."
cd "$DEPLOY_DIR"
if command -v sha256sum &> /dev/null; then
    sha256sum * > "checksums.txt"
elif command -v shasum &> /dev/null; then
    shasum -a 256 * > "checksums.txt"
else
    echo "⚠️ 체크섬 생성 도구를 찾을 수 없습니다."
fi
cd ..

echo ""
echo "🎉 Windows 배포 완료!"
echo "=================================================="
echo "📁 배포 디렉토리: $DEPLOY_DIR"
echo "📦 생성된 파일들:"
ls -la "$DEPLOY_DIR"
echo ""
echo "💡 배포 방법:"
echo "   1. GitHub Releases에 업로드"
echo "   2. 직접 파일 공유"
echo "   3. 웹사이트에 다운로드 링크 제공"
echo ""
echo "🚀 행운을 빕니다!"
