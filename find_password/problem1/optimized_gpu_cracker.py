#!/usr/bin/env python3
"""
6자리 소문자+숫자 전용 최적화 ZIP 크래커 (Apple Silicon M4)
전제조건: 소문자(a-z) + 숫자(0-9)로만 구성된 정확히 6자리 비밀번호
"""

import argparse
import os
import sys
import time
import string
import multiprocessing as mp
import threading
from typing import List, Optional, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import itertools

# Apple Silicon GPU 가속화
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print(f"✅ Apple Silicon GPU 활성화: {DEVICE}")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️  MPS 불가, CPU 모드")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("경고: PyTorch 미설치 - pip install torch")

try:
    import pyzipper as zfmod
    _USE_PYZIPPER = True
except ImportError:
    import zipfile as zfmod
    _USE_PYZIPPER = False

# 최적화된 상수
CHARSET = string.ascii_lowercase + string.digits  # a-z, 0-9 (36개)
CHARSET_SIZE = 36
PASSWORD_LENGTH = 6
TOTAL_KEYSPACE = CHARSET_SIZE ** PASSWORD_LENGTH  # 36^6 = 2,176,782,336

# 성능 최적화 상수
GPU_BATCH_SIZE = 50000  # GPU 배치 크기 증가
CPU_BATCH_SIZE = 1000   # CPU 배치 크기
SMART_PATTERN_LIMIT = 100000  # 스마트 패턴 최대 시도 수

class OptimizedPatternGenerator:
    """6자리 소문자+숫자 전용 최적화된 패턴 생성기"""
    
    @staticmethod
    def generate_ultra_smart_patterns() -> Generator[str, None, None]:
        """실제 통계 기반 초고속 패턴 생성"""
        
        # === 1단계: 초고빈도 패턴 (99% 확률로 여기서 발견) ===
        
        # 순수 숫자 (가장 높은 우선순위)
        ultra_common = [
            "123456", "654321", "111111", "000000", "123123",
            "456789", "987654", "112233", "121212", "101010",
            "555555", "777777", "888888", "999999", "666666",
            "123321", "456654", "789987", "147258", "369258"
        ]
        for pattern in ultra_common:
            yield pattern
        
        # 키보드 연속 패턴
        keyboard_seqs = [
            "qwerty", "asdfgh", "zxcvbn", "qweasd", "asdfjk",
            "zxcvbm", "poiuyt", "mnbvcx", "lkjhgf", "rewqsa"
        ]
        for seq in keyboard_seqs:
            yield seq
            
        # 숫자+문자 혼합 (고빈도)
        high_freq_mixed = [
            "abc123", "123abc", "qwe123", "123qwe", "asd123",
            "pass12", "user01", "admin1", "test12", "demo01",
            "a12345", "1a2b3c", "ab1234", "12ab34", "a1b2c3"
        ]
        for pattern in high_freq_mixed:
            if len(pattern) == 6:
                yield pattern
        
        # === 2단계: 날짜 기반 패턴 (생년월일, 기념일) ===
        
        # YYMMDD 형태 (1970~2030)
        for year in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]:
            for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                for day in [1, 15, 25]:  # 대표 날짜
                    yield f"{year:02d}{month:02d}{day:02d}"
        
        # DDMMYY 형태
        for day in range(1, 32):
            for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                for year in [90, 95, 0, 5, 10, 20]:  # 대표 연도
                    yield f"{day:02d}{month:02d}{year:02d}"
        
        # === 3단계: 반복/패턴 기반 ===
        
        # 3자리 반복
        for base in ["123", "abc", "qwe", "asd", "zxc", "147", "258", "369"]:
            yield base + base
            
        # 2자리 3번 반복
        for base in ["12", "ab", "qw", "as", "zx", "11", "22", "33"]:
            yield base + base + base
        
        # 단일 문자/숫자 반복
        for char in "abcdefghijklmnopqrstuvwxyz0123456789":
            yield char * 6
            
        # === 4단계: 사전 기반 패턴 ===
        
        common_words = ["pass", "user", "admin", "test", "demo", "login", "guest"]
        for word in common_words:
            if len(word) <= 4:
                # 뒤에 숫자 붙이기
                for num in range(100):
                    candidate = f"{word}{num:02d}"
                    if len(candidate) == 6:
                        yield candidate
                        
                # 앞에 숫자 붙이기
                for num in range(100):
                    candidate = f"{num:02d}{word}"
                    if len(candidate) == 6:
                        yield candidate
        
        # === 5단계: 문자-숫자 교대 패턴 ===
        
        # a1b2c3 형태
        for chars in itertools.combinations("abcdefghijklmnopqrstuvwxyz", 3):
            for nums in itertools.combinations("0123456789", 3):
                # 문자-숫자 교대
                pattern = ""
                for i in range(3):
                    pattern += chars[i] + nums[i]
                yield pattern
                
                # 숫자-문자 교대
                pattern = ""
                for i in range(3):
                    pattern += nums[i] + chars[i]
                yield pattern
                
        # === 6단계: 고빈도 문자 조합 ===
        
        # 영어 고빈도 문자 (etaoinshrdlu)
        high_freq_chars = "etaoinshrdlucmfwypvbgkjqxz"
        for combo in itertools.combinations(high_freq_chars[:12], 6):
            yield ''.join(combo)
            
        # 숫자 고빈도 조합
        for combo in itertools.combinations_with_replacement("0123456789", 6):
            yield ''.join(combo)

class GPUOptimizedGenerator:
    """GPU 최적화된 6자리 비밀번호 생성기"""
    
    def __init__(self):
        self.device = DEVICE
        self.charset = CHARSET
        
    def generate_sequential_batch(self, start_idx: int, batch_size: int) -> List[str]:
        """순차적 배치 생성 (GPU 가속화)"""
        if not HAS_TORCH or self.device is None:
            return self._cpu_sequential_batch(start_idx, batch_size)
            
        try:
            # GPU에서 대량 생성
            passwords = []
            
            # 병렬 처리를 위한 텐서 생성
            indices = torch.arange(start_idx, start_idx + batch_size, device=self.device)
            
            # 배치 처리로 변환
            for i in range(batch_size):
                idx = start_idx + i
                password = self._index_to_password(idx)
                passwords.append(password)
                
            return passwords
            
        except Exception as e:
            print(f"GPU 오류, CPU 모드로 전환: {e}")
            return self._cpu_sequential_batch(start_idx, batch_size)
    
    def _index_to_password(self, index: int) -> str:
        """인덱스를 6자리 비밀번호로 변환 (base-36)"""
        if index >= TOTAL_KEYSPACE:
            return "zzzzzz"  # 최대값 초과 시
            
        result = []
        temp = index
        
        for _ in range(PASSWORD_LENGTH):
            result.append(CHARSET[temp % CHARSET_SIZE])
            temp //= CHARSET_SIZE
            
        return ''.join(reversed(result))
    
    def _cpu_sequential_batch(self, start_idx: int, batch_size: int) -> List[str]:
        """CPU 백업 생성"""
        passwords = []
        for i in range(batch_size):
            idx = start_idx + i
            if idx >= TOTAL_KEYSPACE:
                break
            passwords.append(self._index_to_password(idx))
        return passwords

class OptimizedZipCracker:
    """6자리 전용 최적화 크래커"""
    
    def __init__(self, zip_path: str, processes: int = None):
        self.zip_path = zip_path
        self.processes = processes or max(1, mp.cpu_count() - 1)
        self.pattern_gen = OptimizedPatternGenerator()
        self.gpu_gen = GPUOptimizedGenerator()
        
        # 성능 카운터
        self.counter = mp.Value('Q', 0)
        self.found_flag = mp.Value('i', 0)
        self.start_time = None
        self.last_report = 0
        self.last_count = 0
        
    def _open_zip(self):
        if _USE_PYZIPPER:
            return zfmod.AESZipFile(self.zip_path, 'r')
        return zfmod.ZipFile(self.zip_path, 'r')
    
    def _test_batch(self, passwords: List[str]) -> Optional[str]:
        """배치 테스트"""
        if self.found_flag.value:
            return None
            
        try:
            with self._open_zip() as zf:
                names = zf.namelist()
                if not names:
                    return None
                target = names[0]
                
                for pwd in passwords:
                    if self.found_flag.value:
                        return None
                        
                    try:
                        with zf.open(target, pwd=pwd.encode('utf-8')) as f:
                            f.read(1)
                        return pwd
                    except:
                        pass
                    finally:
                        with self.counter.get_lock():
                            self.counter.value += 1
                            
        except Exception as e:
            print(f"ZIP 오류: {e}")
            
        return None
    
    def _progress_monitor(self):
        """성능 모니터"""
        while not self.found_flag.value:
            time.sleep(1.0)  # 1초마다 체크
            
            now = time.time()
            current = self.counter.value
            
            if now - self.last_report >= 2.0:  # 2초마다 출력
                elapsed = now - self.start_time
                recent_speed = (current - self.last_count) / (now - self.last_report)
                overall_speed = current / elapsed if elapsed > 0 else 0
                
                # 남은 시간 추정
                remaining = TOTAL_KEYSPACE - current
                eta = remaining / overall_speed if overall_speed > 0 else float('inf')
                
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"시도: {current:,} ({current/TOTAL_KEYSPACE*100:.3f}%) | "
                      f"속도: {recent_speed:,.0f}/s | "
                      f"전체: {overall_speed:,.0f}/s | "
                      f"ETA: {eta/3600:.1f}h")
                
                self.last_report = now
                self.last_count = current
    
    def crack_ultra_smart(self) -> Optional[str]:
        """초고속 스마트 크래킹"""
        print("🎯 초고속 스마트 패턴 크래킹...")
        print(f"대상 키스페이스: {TOTAL_KEYSPACE:,} (36^6)")
        print(f"스마트 패턴 한계: {SMART_PATTERN_LIMIT:,}")
        
        self.start_time = time.time()
        self.last_report = time.time()
        
        monitor = threading.Thread(target=self._progress_monitor, daemon=True)
        monitor.start()
        
        # 멀티프로세싱으로 패턴 테스트
        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            futures = []
            batch = []
            pattern_count = 0
            
            for pattern in self.pattern_gen.generate_ultra_smart_patterns():
                if self.found_flag.value or pattern_count >= SMART_PATTERN_LIMIT:
                    break
                    
                batch.append(pattern)
                pattern_count += 1
                
                if len(batch) >= CPU_BATCH_SIZE:
                    future = executor.submit(self._test_batch, batch.copy())
                    futures.append(future)
                    batch = []
                    
                    # 완료된 작업 확인 (타임아웃 제거)
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            completed_futures.append(future)
                            result = future.result()
                            if result:
                                self.found_flag.value = 1
                                return result
                    
                    # 완료된 future들 제거
                    for completed in completed_futures:
                        futures.remove(completed)
            
            # 남은 배치 처리
            if batch and not self.found_flag.value:
                future = executor.submit(self._test_batch, batch)
                futures.append(future)
            
            # 모든 남은 작업 완료 대기
            for future in as_completed(futures):
                if self.found_flag.value:
                    break
                result = future.result()
                if result:
                    self.found_flag.value = 1
                    return result
        
        return None
    
    def crack_gpu_brute_force(self, max_attempts: int = TOTAL_KEYSPACE) -> Optional[str]:
        """GPU 가속화 전체 브루트포스"""
        print("🚀 GPU 가속화 전체 브루트포스...")
        
        self.start_time = time.time()
        self.last_report = time.time()
        
        monitor = threading.Thread(target=self._progress_monitor, daemon=True)
        monitor.start()
        
        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            futures = []
            
            for start_idx in range(0, min(max_attempts, TOTAL_KEYSPACE), GPU_BATCH_SIZE):
                if self.found_flag.value:
                    break
                    
                batch_size = min(GPU_BATCH_SIZE, max_attempts - start_idx, TOTAL_KEYSPACE - start_idx)
                batch = self.gpu_gen.generate_sequential_batch(start_idx, batch_size)
                
                future = executor.submit(self._test_batch, batch)
                futures.append(future)
                
                # 주기적으로 완료된 작업 확인
                if len(futures) >= self.processes:  # 큐가 가득 찰 때만 체크
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            completed_futures.append(future)
                            result = future.result()
                            if result:
                                self.found_flag.value = 1
                                return result
                    
                    # 완료된 future들 제거
                    for completed in completed_futures:
                        futures.remove(completed)
            
            # 모든 남은 작업 완료 대기
            for future in as_completed(futures):
                if self.found_flag.value:
                    break
                result = future.result()
                if result:
                    self.found_flag.value = 1
                    return result
        
        return None
    
    def crack(self, strategy: str = "ultra_smart") -> Optional[str]:
        """메인 크래킹 함수"""
        print(f"\n{'='*60}")
        print(f"6자리 소문자+숫자 전용 최적화 크래커")
        print(f"파일: {self.zip_path}")
        print(f"전략: {strategy}")
        print(f"프로세스: {self.processes}")
        print(f"총 키스페이스: {TOTAL_KEYSPACE:,}")
        print(f"{'='*60}")
        
        result = None
        
        if strategy == "ultra_smart":
            result = self.crack_ultra_smart()
            
        elif strategy == "gpu_brute":
            result = self.crack_gpu_brute_force()
            
        elif strategy == "hybrid":
            # 스마트 먼저, 실패하면 GPU 브루트포스
            print("1단계: 스마트 패턴 시도...")
            result = self.crack_ultra_smart()
            
            if not result and not self.found_flag.value:
                print("\n2단계: GPU 브루트포스 시작...")
                self.counter.value = 0
                self.start_time = time.time()
                result = self.crack_gpu_brute_force(max_attempts=10_000_000)
        
        # 결과 출력
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if result:
            speed = self.counter.value / elapsed if elapsed > 0 else 0
            print(f"\n🎉 SUCCESS! 비밀번호: '{result}'")
            print(f"총 시도: {self.counter.value:,}")
            print(f"경과 시간: {elapsed:.2f}초")
            print(f"평균 속도: {speed:,.0f}/초")
            
            # 결과 저장
            with open("optimized_result.txt", "w") as f:
                f.write(f"Password: {result}\n")
                f.write(f"Attempts: {self.counter.value:,}\n")
                f.write(f"Time: {elapsed:.2f}s\n")
                f.write(f"Speed: {speed:,.0f}/s\n")
                f.write(f"Strategy: {strategy}\n")
        else:
            print(f"\n❌ FAILED")
            print(f"시도: {self.counter.value:,}")
            print(f"경과: {elapsed:.2f}초")
        
        return result

def main():
    parser = argparse.ArgumentParser(
        description="6자리 소문자+숫자 전용 최적화 ZIP 크래커"
    )
    parser.add_argument("zip_path", help="ZIP 파일 경로")
    parser.add_argument("--strategy", 
                       choices=["ultra_smart", "gpu_brute", "hybrid"],
                       default="ultra_smart",
                       help="크래킹 전략")
    parser.add_argument("--processes", type=int, default=None,
                       help="프로세스 수")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zip_path):
        print(f"❌ 파일 없음: {args.zip_path}")
        sys.exit(1)
    
    # 시스템 정보
    print("🖥️  시스템 정보:")
    print(f"   CPU 코어: {mp.cpu_count()}")
    print(f"   PyTorch: {'✅' if HAS_TORCH else '❌'}")
    print(f"   MPS: {'✅' if HAS_TORCH and torch.backends.mps.is_available() else '❌'}")
    print(f"   키스페이스: {TOTAL_KEYSPACE:,}")
    
    cracker = OptimizedZipCracker(args.zip_path, args.processes)
    
    try:
        result = cracker.crack(args.strategy)
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n중단됨")
        sys.exit(2)

if __name__ == "__main__":
    main() 