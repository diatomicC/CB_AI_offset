# 🚀 GitHub Actions CI/CD 가이드

## 📋 개요

이 가이드는 GitHub Actions를 사용하여 CMYK Analyzer를 자동으로 빌드하고 배포하는 방법을 설명합니다.

## 🔧 설정 방법

### 1. GitHub 저장소 설정

1. **GitHub에 코드 푸시**
   ```bash
   git add .
   git commit -m "Add GitHub Actions workflows"
   git push origin main
   ```

2. **Actions 탭 확인**
   - GitHub 저장소에서 "Actions" 탭 클릭
   - 워크플로우가 자동으로 실행되는지 확인

### 2. 워크플로우 파일 구조

```
.github/
└── workflows/
    ├── build.yml              # 전체 플랫폼 빌드
    └── build-windows.yml      # Windows 전용 빌드
```

## 🎯 워크플로우 트리거

### 자동 실행 조건

- **Push**: `main`, `develop` 브랜치에 푸시
- **Tags**: `v1.0.0` 같은 태그 생성
- **Pull Request**: `main`, `develop` 브랜치로 PR
- **Manual**: GitHub Actions 페이지에서 수동 실행

### 수동 실행 방법

1. GitHub 저장소 → Actions 탭
2. "Build Windows Executable" 워크플로우 선택
3. "Run workflow" 버튼 클릭
4. 브랜치 선택 후 실행

## 🏗️ 빌드 프로세스

### Windows 빌드 과정

1. **환경 설정**
   - Windows 최신 러너 사용
   - Python 3.11 설치
   - 의존성 설치

2. **실행파일 생성**
   ```bash
   pyinstaller --onefile --windowed --icon=app_icon.ico --name=CMYK_Analyzer run_gui.py
   ```

3. **배포 패키지 생성**
   - 실행파일과 아이콘 포함
   - ZIP 파일로 압축

4. **아티팩트 업로드**
   - GitHub Actions 아티팩트로 저장
   - 30일간 보관

## 📦 결과물

### 생성되는 파일들

- **`CMYK_Analyzer.exe`**: Windows 실행파일
- **`CMYK_Analyzer_Windows.zip`**: 배포 패키지
- **`app_icon.ico`**: 애플리케이션 아이콘

### 다운로드 방법

1. GitHub Actions 실행 완료 후
2. "Artifacts" 섹션에서 "windows-executable" 클릭
3. ZIP 파일 다운로드

## 🚨 문제 해결

### 일반적인 오류

1. **의존성 설치 실패**
   - `requirements.txt` 확인
   - Python 버전 호환성 체크

2. **PyInstaller 오류**
   - 숨겨진 import 확인
   - 파일 경로 문제 해결

3. **메모리 부족**
   - Windows 러너 사양 확인
   - 빌드 최적화

### 디버깅 방법

1. **로그 확인**
   - Actions 탭에서 실행 로그 확인
   - 각 단계별 성공/실패 상태 확인

2. **로컬 테스트**
   - 동일한 환경에서 로컬 빌드 테스트
   - 의존성 충돌 확인

## 🔄 자동화 워크플로우

### 릴리스 자동화

1. **태그 생성**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **자동 릴리스**
   - GitHub Actions가 자동으로 릴리스 생성
   - 모든 플랫폼 실행파일 포함
   - 릴리스 노트 자동 생성

### 브랜치 보호

1. **main 브랜치 보호**
   - PR 리뷰 필수
   - 상태 체크 통과 필수
   - 자동 빌드 성공 확인

## 📊 모니터링

### 빌드 상태 확인

- **Actions 탭**: 실시간 빌드 상태
- **Notifications**: 이메일/앱 알림
- **Badge**: README에 빌드 상태 표시

### 성능 최적화

- **캐싱**: pip 캐시, PyInstaller 캐시
- **병렬 실행**: 여러 Python 버전 동시 빌드
- **조건부 실행**: 변경된 파일만 빌드

## 💡 고급 기능

### 매트릭스 빌드

- **Python 버전**: 3.9, 3.10, 3.11
- **플랫폼**: Windows, macOS, Linux
- **자동 테스트**: 각 환경에서 실행 테스트

### 보안

- **Secrets**: 민감한 정보 보호
- **Code signing**: Windows 실행파일 서명
- **Virus scanning**: 바이러스 검사

## 📞 지원

문제가 발생하거나 추가 도움이 필요한 경우:

1. **GitHub Issues**: 저장소에 이슈 생성
2. **Actions 로그**: 빌드 실패 원인 분석
3. **커뮤니티**: GitHub Actions 관련 포럼

---

© 2025 CMYK Analyzer Team
