# macOS 설치 및 사용 가이드

## 🎯 CMYK Registration & Tilt Analyzer

이 가이드는 macOS에서 CMYK Analyzer를 설치하고 사용하는 방법을 설명합니다.

## 📦 설치 방법

### 방법 1: 자동 빌드 (권장)

1. **터미널을 열고 프로젝트 디렉토리로 이동:**
   ```bash
   cd /path/to/project3
   ```

2. **빌드 스크립트 실행:**
   ```bash
   ./build.sh
   ```

3. **빌드 완료 후 앱 번들을 Applications 폴더로 이동:**
   ```bash
   cp -r dist/CMYK_Analyzer.app /Applications/
   ```

### 방법 2: Python 스크립트 사용

1. **Python 빌드 스크립트 실행:**
   ```bash
   python build_macos.py
   ```

2. **앱 번들을 Applications 폴더로 이동**

### 방법 3: PyInstaller 직접 사용

1. **PyInstaller 설치:**
   ```bash
   pip install pyinstaller
   ```

2. **앱 빌드:**
   ```bash
   pyinstaller CMYK_Analyzer.spec
   ```

## 🚀 실행 방법

### 방법 1: Applications 폴더에서 실행
1. Finder 열기
2. Applications 폴더로 이동
3. "CMYK_Analyzer" 앱 더블클릭

### 방법 2: 터미널에서 실행
```bash
open /Applications/CMYK_Analyzer.app
```

### 방법 3: 직접 실행
```bash
./dist/CMYK_Analyzer.app/Contents/MacOS/CMYK_Analyzer
```

## 🔧 문제 해결

### 앱이 실행되지 않는 경우

1. **보안 설정 확인:**
   - System Preferences > Security & Privacy > General
   - "Allow apps downloaded from" 설정 확인
   - 필요한 경우 "Open Anyway" 클릭

2. **권한 문제:**
   ```bash
   chmod +x dist/CMYK_Analyzer.app/Contents/MacOS/CMYK_Analyzer
   ```

3. **의존성 문제:**
   - Python 환경 확인
   - requirements.txt의 패키지들이 설치되어 있는지 확인

### 빌드 오류가 발생하는 경우

1. **PyInstaller 재설치:**
   ```bash
   pip uninstall pyinstaller
   pip install pyinstaller
   ```

2. **Python 버전 확인:**
   - Python 3.8 이상 필요
   ```bash
   python --version
   ```

3. **가상환경 사용 권장:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 📁 파일 구조

빌드 후 생성되는 파일들:

```
dist/
├── CMYK_Analyzer              # 단일 실행파일
└── CMYK_Analyzer.app/         # macOS 앱 번들
    └── Contents/
        ├── MacOS/
        │   └── CMYK_Analyzer   # 실행 파일
        ├── Resources/
        │   └── AppIcon.icns    # 앱 아이콘
        └── Info.plist          # 앱 정보
```

## 🎨 아이콘 커스터마이징

1. **아이콘 파일 준비:**
   - `.icns` 형식의 파일 필요
   - 다양한 해상도 포함 권장 (16x16, 32x32, 128x128, 256x256, 512x512)

2. **아이콘 교체:**
   - `MyIcon.icns` 파일을 원하는 아이콘으로 교체
   - 빌드 스크립트 재실행

## 📝 추가 정보

- **지원 OS:** macOS 10.15 (Catalina) 이상
- **아키텍처:** Intel 및 Apple Silicon (Universal Binary)
- **Python 버전:** 3.8 이상
- **의존성:** PySide6, OpenCV, NumPy, Pillow 등

## 🆘 도움말

문제가 발생하거나 추가 도움이 필요한 경우:

1. 프로젝트 README.md 확인
2. 빌드 로그 확인
3. Python 환경 및 의존성 상태 확인
4. GitHub Issues 검색 또는 새 이슈 생성
