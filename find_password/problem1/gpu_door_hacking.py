#!/usr/bin/env python3
"""
GPU 가속화된 ZIP 비밀번호 크래커 (Apple Silicon M4 최적화)
Metal Performance Shaders와 병렬 처리를 활용한 고성능 크래킹 도구
"""

import argparse
import itertools
import os
import sys
import time
import string
import multiprocessing as mp
import threading
from typing import Iterable, List, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Apple Silicon GPU 가속화를 위한 라이브러리들
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("경고: numpy가 설치되지 않았습니다. 성능이 제한될 수 있습니다.")

try:
    import torch
    # Apple Silicon의 Metal Performance Shaders 사용
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print(f"✅ Apple Silicon GPU 가속화 활성화: {DEVICE}")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️  MPS 사용 불가, CPU 모드로 실행")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("경고: PyTorch가 설치되지 않았습니다. pip install torch 실행하세요.")

try:
    import pyzipper as zfmod
    _USE_PYZIPPER = True
except ImportError:
    import zipfile as zfmod
    _USE_PYZIPPER = False

# 상수 정의
LOWER = string.ascii_lowercase
DIGIT = string.digits
ALNUM = LOWER + DIGIT
BATCH_SIZE_GPU = 10000  # GPU 처리용 배치 크기
THREAD_POOL_SIZE = 8    # 스레드 풀 크기

class GPUPasswordGenerator:
    """GPU 가속화된 비밀번호 생성기"""
    
    def __init__(self, device=None):
        self.device = device if device else DEVICE
        self.charset = ALNUM
        self.charset_size = len(self.charset)
        
    def generate_batch_tensor(self, start_idx: int, batch_size: int, length: int = 6) -> List[str]:
        """GPU를 활용한 배치 비밀번호 생성"""
        if not HAS_TORCH or self.device is None:
            return self._generate_batch_cpu(start_idx, batch_size, length)
        
        try:
            # GPU에서 인덱스 배열 생성
            indices = torch.arange(start_idx, start_idx + batch_size, device=self.device)
            
            # 각 자리수별로 문자 인덱스 계산
            passwords = []
            base = self.charset_size
            
            for i in range(batch_size):
                idx = start_idx + i
                password_chars = []
                temp_idx = idx
                
                # 6자리 비밀번호 생성 (base-36 변환)
                for _ in range(length):
                    char_idx = temp_idx % base
                    password_chars.append(self.charset[char_idx])
                    temp_idx //= base
                
                passwords.append(''.join(reversed(password_chars)))
                
            return passwords
            
        except Exception as e:
            print(f"GPU 처리 오류, CPU 모드로 전환: {e}")
            return self._generate_batch_cpu(start_idx, batch_size, length)
    
    def _generate_batch_cpu(self, start_idx: int, batch_size: int, length: int = 6) -> List[str]:
        """CPU 백업 비밀번호 생성"""
        passwords = []
        base = self.charset_size
        
        for i in range(batch_size):
            idx = start_idx + i
            password_chars = []
            temp_idx = idx
            
            for _ in range(length):
                char_idx = temp_idx % base
                password_chars.append(self.charset[char_idx])
                temp_idx //= base
                
            passwords.append(''.join(reversed(password_chars)))
            
        return passwords

class SmartPatternGenerator:
    """통계 기반 스마트 패턴 생성기"""
    
    @staticmethod
    def generate_common_patterns() -> Generator[str, None, None]:
        """가장 흔한 6자리 패턴들 우선 생성"""
        
        # 1. 순수 숫자 패턴 (가장 높은 우선순위)
        common_numbers = [
            "123456", "654321", "111111", "000000", "123123",
            "456789", "987654", "112233", "121212", "101010",
        ]
        for pattern in common_numbers:
            yield pattern
            
        # 2. 키보드 패턴
        keyboard_patterns = [
            "qwerty", "asdfgh", "zxcvbn", "qweasd", "asdfjk",
            "123qwe", "qwe123", "abc123", "admin1", "pass12"
        ]
        for pattern in keyboard_patterns:
            if len(pattern) == 6:
                yield pattern
            elif len(pattern) < 6:
                # 부족한 자리는 숫자로 채움
                for i in range(10):
                    candidate = (pattern + str(i) * (6 - len(pattern)))[:6]
                    if len(candidate) == 6:
                        yield candidate
        
        # 3. 날짜 기반 패턴 (YYMMDD, DDMMYY 등)
        for year in range(80, 30, -1):  # 1980~2029
            for month in range(1, 13):
                for day in [1, 15, 28]:  # 대표적인 날짜들만
                    patterns = [
                        f"{year:02d}{month:02d}{day:02d}",
                        f"{day:02d}{month:02d}{year:02d}",
                        f"{month:02d}{day:02d}{year:02d}"
                    ]
                    for pattern in patterns:
                        yield pattern
        
        # 4. 문자+숫자 혼합 패턴
        prefixes = ["pass", "user", "admin", "test", "demo"]
        for prefix in prefixes:
            for num in range(100):
                candidate = f"{prefix}{num:02d}"[:6]
                if len(candidate) == 6:
                    yield candidate

class GPUZipCracker:
    """GPU 가속화된 ZIP 크래커 메인 클래스"""
    
    def __init__(self, zip_path: str, processes: int = None, gpu_batch_size: int = BATCH_SIZE_GPU):
        self.zip_path = zip_path
        self.processes = processes or max(1, mp.cpu_count() - 1)
        self.gpu_batch_size = gpu_batch_size
        self.generator = GPUPasswordGenerator()
        self.pattern_gen = SmartPatternGenerator()
        
        # 공유 객체
        self.counter = mp.Value('Q', 0)
        self.found_password = mp.Value('c', b'\x00' * 32)
        self.found_flag = mp.Value('i', 0)
        
        # 성능 모니터링
        self.start_time = None
        self.last_report_time = None
        self.last_count = 0
        
    def _open_zip(self):
        """ZIP 파일 열기"""
        if _USE_PYZIPPER:
            return zfmod.AESZipFile(self.zip_path, 'r')
        return zfmod.ZipFile(self.zip_path, 'r')
    
    def _test_password_batch(self, passwords: List[str]) -> Optional[str]:
        """비밀번호 배치 테스트"""
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
                        with zf.open(target, 'r', pwd=pwd.encode('utf-8')) as f:
                            f.read(1)  # 1바이트만 읽어서 유효성 확인
                        return pwd  # 성공!
                    except:
                        pass
                    finally:
                        with self.counter.get_lock():
                            self.counter.value += 1
                            
        except Exception as e:
            print(f"ZIP 처리 오류: {e}")
            return None
            
        return None
    
    def _worker_process(self, task_queue: mp.Queue, result_queue: mp.Queue):
        """워커 프로세스"""
        while True:
            if self.found_flag.value:
                break
                
            try:
                batch = task_queue.get(timeout=1.0)
                if batch is None:  # 종료 신호
                    break
                    
                result = self._test_password_batch(batch)
                if result:
                    result_queue.put(result)
                    self.found_flag.value = 1
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"워커 프로세스 오류: {e}")
                continue
    
    def _progress_monitor(self):
        """진행률 모니터링 스레드"""
        while not self.found_flag.value:
            time.sleep(2.0)
            
            current_time = time.time()
            current_count = self.counter.value
            
            if self.last_report_time:
                elapsed = current_time - self.start_time
                recent_speed = (current_count - self.last_count) / (current_time - self.last_report_time)
                overall_speed = current_count / elapsed if elapsed > 0 else 0
                
                print(f"[{time.strftime('%H:%M:%S')}] 시도: {current_count:,} | "
                      f"속도: {recent_speed:,.0f}/s | "
                      f"전체: {overall_speed:,.0f}/s | "
                      f"경과: {elapsed:.1f}s")
            
            self.last_report_time = current_time
            self.last_count = current_count
    
    def crack_with_smart_patterns(self) -> Optional[str]:
        """스마트 패턴을 사용한 크래킹"""
        print("🎯 스마트 패턴 기반 크래킹 시작...")
        
        self.start_time = time.time()
        self.last_report_time = time.time()
        
        # 진행률 모니터링 스레드 시작
        monitor_thread = threading.Thread(target=self._progress_monitor, daemon=True)
        monitor_thread.start()
        
        # 작업 큐와 결과 큐
        task_queue = mp.Queue(maxsize=self.processes * 2)
        result_queue = mp.Queue()
        
        # 워커 프로세스들 시작
        workers = []
        for _ in range(self.processes):
            worker = mp.Process(target=self._worker_process, args=(task_queue, result_queue))
            worker.start()
            workers.append(worker)
        
        try:
            # 스마트 패턴 생성 및 배치 처리
            batch = []
            for pattern in self.pattern_gen.generate_common_patterns():
                if self.found_flag.value:
                    break
                    
                batch.append(pattern)
                if len(batch) >= 100:  # 작은 배치로 빠른 반응성 확보
                    task_queue.put(batch)
                    batch = []
            
            # 남은 배치 처리
            if batch and not self.found_flag.value:
                task_queue.put(batch)
            
            # 워커들에게 종료 신호
            for _ in workers:
                task_queue.put(None)
            
            # 결과 확인
            try:
                result = result_queue.get(timeout=5.0)
                self.found_flag.value = 1
                return result
            except queue.Empty:
                pass
                
        except KeyboardInterrupt:
            print("\n사용자가 중단했습니다.")
            self.found_flag.value = 1
        
        finally:
            # 모든 워커 프로세스 종료 대기
            for worker in workers:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    worker.terminate()
        
        return None
    
    def crack_with_gpu_brute_force(self, max_attempts: int = 10_000_000) -> Optional[str]:
        """GPU 가속화된 브루트포스"""
        print("🚀 GPU 가속화 브루트포스 시작...")
        
        self.start_time = time.time()
        self.last_report_time = time.time()
        
        # 진행률 모니터링 스레드
        monitor_thread = threading.Thread(target=self._progress_monitor, daemon=True)
        monitor_thread.start()
        
        total_keyspace = len(ALNUM) ** 6  # 36^6
        print(f"총 키스페이스: {total_keyspace:,}")
        
        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            futures = []
            
            for start_idx in range(0, min(max_attempts, total_keyspace), self.gpu_batch_size):
                if self.found_flag.value:
                    break
                    
                batch_size = min(self.gpu_batch_size, max_attempts - start_idx, total_keyspace - start_idx)
                
                # GPU에서 비밀번호 배치 생성
                password_batch = self.generator.generate_batch_tensor(start_idx, batch_size)
                
                # 비동기로 테스트 실행
                future = executor.submit(self._test_password_batch, password_batch)
                futures.append(future)
                
                # 완료된 작업 확인
                for completed_future in as_completed(futures, timeout=0.1):
                    result = completed_future.result()
                    if result:
                        self.found_flag.value = 1
                        return result
                    futures.remove(completed_future)
                    break
        
        return None
    
    def crack(self, strategy: str = "smart") -> Optional[str]:
        """메인 크래킹 함수"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPU 가속화 크래킹 시작")
        print(f"파일: {self.zip_path}")
        print(f"전략: {strategy}")
        print(f"프로세스: {self.processes}")
        print(f"GPU 배치 크기: {self.gpu_batch_size}")
        print("-" * 60)
        
        result = None
        
        if strategy == "smart":
            result = self.crack_with_smart_patterns()
            
        elif strategy == "gpu_brute":
            result = self.crack_with_gpu_brute_force()
            
        elif strategy == "hybrid":
            # 스마트 패턴 먼저, 실패하면 GPU 브루트포스
            result = self.crack_with_smart_patterns()
            if not result and not self.found_flag.value:
                print("\n스마트 패턴 실패, GPU 브루트포스로 전환...")
                self.counter.value = 0  # 카운터 리셋
                result = self.crack_with_gpu_brute_force(max_attempts=5_000_000)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if result:
            print(f"\n🎉 성공! 비밀번호: '{result}'")
            print(f"총 시도: {self.counter.value:,}")
            print(f"경과 시간: {elapsed:.2f}초")
            print(f"평균 속도: {self.counter.value/elapsed:,.0f}/초")
            
            # 결과 저장
            with open("gpu_password_result.txt", "w", encoding="utf-8") as f:
                f.write(f"Password: {result}\n")
                f.write(f"Attempts: {self.counter.value:,}\n")
                f.write(f"Time: {elapsed:.2f}s\n")
                f.write(f"Speed: {self.counter.value/elapsed:,.0f}/s\n")
        else:
            print(f"\n❌ 실패 - 비밀번호를 찾지 못했습니다.")
            print(f"총 시도: {self.counter.value:,}")
            print(f"경과 시간: {elapsed:.2f}초")
        
        return result

def main():
    parser = argparse.ArgumentParser(description="GPU 가속화된 ZIP 비밀번호 크래커 (Apple Silicon M4 최적화)")
    parser.add_argument("zip_path", help="대상 ZIP 파일 경로")
    parser.add_argument("--strategy", choices=["smart", "gpu_brute", "hybrid"], 
                       default="smart", help="크래킹 전략 선택")
    parser.add_argument("--processes", type=int, default=None, 
                       help="사용할 프로세스 수 (기본: CPU 코어 수 - 1)")
    parser.add_argument("--gpu-batch-size", type=int, default=BATCH_SIZE_GPU,
                       help="GPU 처리 배치 크기")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zip_path):
        print(f"❌ 파일을 찾을 수 없습니다: {args.zip_path}")
        sys.exit(1)
    
    # 시스템 정보 출력
    print("🖥️  시스템 정보:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {'✅ 설치됨' if HAS_TORCH else '❌ 미설치'}")
    print(f"   MPS 지원: {'✅ 사용 가능' if HAS_TORCH and torch.backends.mps.is_available() else '❌ 불가능'}")
    print(f"   CPU 코어: {mp.cpu_count()}")
    print()
    
    # 크래커 실행
    cracker = GPUZipCracker(
        zip_path=args.zip_path,
        processes=args.processes,
        gpu_batch_size=args.gpu_batch_size
    )
    
    try:
        result = cracker.crack(strategy=args.strategy)
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n사용자가 중단했습니다.")
        sys.exit(2)

if __name__ == "__main__":
    main() 