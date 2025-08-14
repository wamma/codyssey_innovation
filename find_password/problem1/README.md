# ZIP 비밀번호 크래커 (Door Hacking Tool)

6자리 소문자/숫자 조합의 ZIP 파일 비밀번호를 찾는 고성능 브루트포스 도구입니다.

## 개요

이 도구는 암호화된 ZIP 파일의 비밀번호를 찾기 위해 다양한 전략을 사용하는 멀티프로세싱 기반의 크래킹 도구입니다. 6자리 길이의 소문자와 숫자 조합으로 이루어진 비밀번호를 대상으로 합니다.

## 주요 특징

- **멀티프로세싱**: CPU 코어를 최대한 활용하여 병렬 처리
- **다양한 전략**: 사전 기반, 마스크 기반, 하이브리드 접근법
- **실시간 진행률**: 시도 횟수, 속도, 경과 시간 실시간 표시
- **스마트 중단**: 비밀번호 발견 시 모든 프로세스 즉시 중단
- **결과 저장**: 발견된 비밀번호를 `password.txt` 파일에 자동 저장

## 의존성

### 필수 라이브러리
- Python 3.6 이상
- 기본 라이브러리: `argparse`, `itertools`, `os`, `sys`, `time`, `string`, `multiprocessing`

### 선택적 라이브러리
- `pyzipper`: AES 암호화된 ZIP 파일 지원 (권장)
  ```bash
  pip install pyzipper
  ```
  > 설치되지 않은 경우 기본 `zipfile` 모듈 사용

## 사용법

### 기본 사용법
```bash
python door_hacking.py <ZIP_파일_경로>
```

### 전략별 사용법

#### 1. 하이브리드 전략 (기본값)
```bash
python door_hacking.py emergency_storage_key.zip
```
- 빈도 높은 패턴 + 마스크 패턴 조합

#### 2. 사전 기반 전략
```bash
python door_hacking.py emergency_storage_key.zip --strategy dict
```
- 일반적인 패턴들을 우선적으로 시도

#### 3. 마스크 기반 전략
```bash
python door_hacking.py emergency_storage_key.zip --strategy mask --mask ?d?d?d?d?d?d
```
- 특정 패턴만 집중적으로 시도

### 고급 옵션

```bash
python door_hacking.py emergency_storage_key.zip \
    --strategy hybrid \
    --mask ?l?l?d?d?d?d \
    --processes 8 \
    --batch-size 5000 \
    --progress-interval 1.0
```

## 전략 설명

### 1. 사전 기반 (dict)
다음 순서로 패턴을 시도합니다:
1. **6자리 숫자**: `000000` ~ `999999`
2. **빈도 높은 알파벳**: 영어에서 자주 사용되는 문자 조합
3. **혼합 패턴**: 문자와 숫자의 다양한 조합
4. **접두사 + 숫자**: `pass01`, `qwer12`, `admin99` 등

### 2. 마스크 기반 (mask)
특정 패턴을 정의하여 체계적으로 시도:
- `?d`: 숫자 (0-9)
- `?l`: 소문자 (a-z)
- `?a`: 알파뉴메릭 (a-z, 0-9)

**예시:**
- `?d?d?d?d?d?d`: 6자리 숫자만
- `?l?l?l?l?l?l`: 6자리 소문자만
- `?l?l?d?d?d?d`: 앞 2자리 문자 + 뒤 4자리 숫자

### 3. 하이브리드 (hybrid)
사전 기반 + 마스크 기반을 순차적으로 실행

## 성능 최적화

### 프로세스 수 조정
```bash
--processes 16  # CPU 코어 수에 맞게 조정
```

### 배치 크기 조정
```bash
--batch-size 5000  # 메모리와 성능의 균형점
```

### 진행률 표시 간격
```bash
--progress-interval 1.0  # 1초마다 진행률 표시
```

## 출력 예시

```
[2024-01-15 14:30:25] 시작 :: 파일='emergency_storage_key.zip', 전략='hybrid', 프로세스=8
[2024-01-15 14:30:27] 시도=50,000 최근속도≈25,000/s 전체속도≈25,000/s 경과=2.0s
[2024-01-15 14:30:29] 시도=100,000 최근속도≈25,000/s 전체속도≈25,000/s 경과=4.0s
[2024-01-15 14:30:31] 성공! 비밀번호='abc123', 총 시도=123,456, 경과=6.2s
```

## 작동 원리

1. **초기화**: 지정된 수의 워커 프로세스 생성
2. **후보 생성**: 선택된 전략에 따라 비밀번호 후보 생성
3. **배치 처리**: 후보들을 배치 단위로 묶어서 각 프로세스에 분배
4. **병렬 테스트**: 각 프로세스가 독립적으로 비밀번호 시도
5. **결과 수집**: 성공 시 즉시 모든 프로세스 중단 및 결과 저장

## 주의사항

- **리소스 사용**: 높은 CPU 사용률로 인해 시스템이 느려질 수 있습니다
- **시간 소요**: 전체 키스페이스(36^6 = 2,176,782,336) 탐색 시 상당한 시간이 소요될 수 있습니다

## 문제 해결

### pyzipper 설치 오류
```bash
# macOS
brew install libzip
pip install pyzipper

# Ubuntu/Debian
sudo apt-get install libzip-dev
pip install pyzipper
```

### 메모리 부족
배치 크기를 줄여보세요:
```bash
--batch-size 1000
```

### 성능 최적화
- SSD 사용 권장
- 충분한 RAM 확보
- 백그라운드 프로그램 최소화

## 라이선스

이 도구는 교육 및 연구 목적으로 제공됩니다. 합법적인 용도로만 사용하시기 바랍니다. 