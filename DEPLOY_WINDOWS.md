# Windows 배포 가이드

## 🪟 CMYK Registration & Tilt Analyzer Windows 실행파일

이 가이드는 Windows에서 CMYK Analyzer를 빌드하고 배포하는 방법을 설명합니다.

## 📦 빌드 방법

### 방법 1: Python 스크립트 사용 (권장)

```bash
python build_windows.py
```

### 방법 2: PyInstaller 직접 사용

```bash
pyinstaller --onefile --windowed --icon=app_icon.ico run_gui.py
```

### 방법 3: spec 파일 사용

```bash
pyinstaller CMYK_Analyzer_Windows.spec
```

## 🚀 배포 방법

### 자동 배포 스크립트

```bash
./deploy_windows.sh
```

### 수동 배포

1. **ZIP 파일 생성:**
   ```bash
   cd dist
   zip -r CMYK_Analyzer_Windows_v1.0.0.zip CMYK_Analyzer.exe
   ```

2. **설치 프로그램 생성 (NSIS 필요):**
   - NSIS 설치: https://nsis.sourceforge.io/Download
   - `build_windows.py`가 자동으로 NSIS 스크립트 생성

## 📁 배포 파일 구성

```
deploy_windows_1.0.0/
├── CMYK_Analyzer_1.0.0.exe      # Windows 실행파일
├── app_icon.ico                  # 애플리케이션 아이콘
├── CMYK_Analyzer_Windows_1.0.0.zip  # 압축파일
├── CMYK_Analyzer_Setup.exe      # 설치 프로그램 (NSIS)
├── README.txt                    # 설치 가이드
└── checksums.txt                 # 파일 무결성 검증
```

## 🔧 설치 방법

### 방법 1: 직접 실행
- `CMYK_Analyzer_1.0.0.exe` 더블클릭

### 방법 2: 설치 프로그램 (권장)
- `CMYK_Analyzer_Setup.exe` 실행
- 시작 메뉴와 바탕화면에 바로가기 자동 생성

### 방법 3: ZIP 압축 해제
- `CMYK_Analyzer_Windows_1.0.0.zip` 압축 해제
- 원하는 위치에 배치하여 사용

## 📋 시스템 요구사항

- **OS**: Windows 10 이상 (64비트)
- **프레임워크**: .NET Framework 4.7.2 이상
- **메모리**: 최소 4GB RAM
- **디스크**: 500MB 이상의 여유 공간

## 🚨 문제 해결

### 실행이 안 되는 경우
1. **바이러스 백신**: 실행파일을 예외 처리
2. **Windows Defender**: 스마트스크린에서 "추가 정보" → "실행" 클릭
3. **권한 문제**: 관리자 권한으로 실행

### DLL 오류
- **Visual C++ Redistributable** 설치
- **Microsoft Visual C++ 2015-2022 Redistributable** 다운로드

### 화면이 안 보이는 경우
- **Windows 호환성 모드** 설정
- **DPI 설정** 확인

## 🌐 배포 플랫폼

### GitHub Releases
1. GitHub 저장소에 업로드
2. Releases 섹션에서 새 릴리스 생성
3. 사용자들이 직접 다운로드

### 직접 공유
- 이메일, USB, 클라우드 스토리지
- 회사 내부 네트워크
- 웹사이트 다운로드 섹션

## 📝 배포 체크리스트

- [ ] Windows 실행파일 빌드 완료
- [ ] 아이콘 파일 포함
- [ ] 설치 프로그램 생성 (선택사항)
- [ ] ZIP 파일 생성
- [ ] README 파일 작성
- [ ] 체크섬 생성
- [ ] 테스트 완료
- [ ] 배포 파일 업로드

## 🆘 지원

문제가 발생하거나 추가 도움이 필요한 경우:

1. 프로젝트 README.md 확인
2. 빌드 로그 확인
3. Python 환경 및 의존성 상태 확인
4. GitHub Issues 검색 또는 새 이슈 생성

## 📞 연락처

- **GitHub Issues**: [저장소 URL]
- **이메일**: [이메일 주소]
- **문서**: [문서 URL]

---

© 2025 CMYK Analyzer Team
