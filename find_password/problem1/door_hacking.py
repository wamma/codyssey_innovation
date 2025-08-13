# -*- coding: utf-8 -*-
"""
door_hacking.py
---------------
Zip(ZipCrypto 또는 AES)으로 암호화된 파일의 6자리 소문자/숫자 비밀번호를
사전/마스크/병렬 브루트포스로 시도하는 스크립트.
"""

import argparse
import itertools
import os
import sys
import time
import string
import multiprocessing as mp
from typing import Iterable, List, Optional

# AES 지원을 위해 pyzipper가 있으면 사용, 없으면 표준 zipfile 사용
try:
    import pyzipper as zfmod
    _USE_PYZIPPER = True
except Exception:
    import zipfile as zfmod
    _USE_PYZIPPER = False

LOWER = string.ascii_lowercase
DIGIT = string.digits
ALNUM = LOWER + DIGIT

# 전역 변수 (worker에서 접근)
g_counter = None
g_found_event = None

def init_worker(counter_, found_event_):
    """Pool 워커 초기화 함수"""
    global g_counter, g_found_event
    g_counter = counter_
    g_found_event = found_event_

def human_int(n: int) -> str:
    return f"{n:,}"

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _open_zip(zip_path: str):
    if _USE_PYZIPPER:
        return zfmod.AESZipFile(zip_path, 'r')
    return zfmod.ZipFile(zip_path, 'r')

def _test_password(args) -> Optional[str]:
    zip_path, candidates = args
    if g_found_event.is_set():
        return None
    try:
        with _open_zip(zip_path) as zf:
            names = zf.namelist()
            if not names:
                return None
            target = names[0]
            for pwd in candidates:
                if g_found_event.is_set():
                    return None
                try:
                    with zf.open(target, 'r', pwd=pwd.encode('utf-8')) as f:
                        f.read(1)
                    g_found_event.set()
                    return pwd
                except:
                    pass
                finally:
                    with g_counter.get_lock():
                        g_counter.value += 1
    except Exception as e:
        print("ZIP 열기 실패:", e)
        return None
    return None

def chunker(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def generate_from_mask(mask: str) -> Iterable[str]:
    pools: List[str] = []
    i = 0
    while i < len(mask):
        if mask[i] == '?' and i + 1 < len(mask):
            t = mask[i + 1]
            if t == 'd':
                pools.append(DIGIT)
            elif t == 'l':
                pools.append(LOWER)
            elif t == 'a':
                pools.append(ALNUM)
            else:
                raise ValueError(f"알 수 없는 마스크 토큰 '?{t}'")
            i += 2
        else:
            raise ValueError("마스크는 '?d', '?l', '?a' 토큰만 지원합니다.")
    for tup in itertools.product(*pools):
        yield ''.join(tup)

def top_patterns_wordlist() -> Iterable[str]:
    # 000000 ~ 999999
    for n in range(0, 1_000_000):
        yield f"{n:06d}"
    # 빈도 높은 알파벳 조합
    freq_order = "etaoinshrdlucmfwypvbgkjqxz"
    top = freq_order[:12]
    for tup in itertools.product(top, repeat=6):
        yield ''.join(tup)
    # 혼합 패턴
    mixed_masks = [
        "?l?l?d?d?d?d",
        "?d?d?l?l?d?d",
        "?d?d?d?d?l?l",
        "?l?l?l?d?d?d",
        "?d?l?l?d?d?l",
        "?l?d?l?d?l?d",
    ]
    for m in mixed_masks:
        for w in generate_from_mask(m):
            yield w
    # prefix + 숫자
    prefixes = ["pass", "qwer", "asdf", "zxcv", "abc", "admin"]
    suffixes = [f"{i:02d}" for i in range(0, 100)]
    for p in prefixes:
        if len(p) >= 6:
            yield p[:6]
        else:
            for s in suffixes:
                cand = (p + s)[:6]
                if len(cand) == 6:
                    yield cand

def unlock_zip(zip_path: str,
               strategy: str = "hybrid",
               mask: Optional[str] = None,
               processes: Optional[int] = None,
               batch_size: int = 2000,
               progress_interval: float = 2.0) -> Optional[str]:

    if processes is None or processes <= 0:
        processes = max(1, mp.cpu_count() - 1)

    print(f"[{now_str()}] 시작 :: 파일='{zip_path}', 전략='{strategy}', 프로세스={processes}")
    t0 = time.time()

    candidate_iters: List[Iterable[str]] = []
    if strategy == "dict":
        candidate_iters = [top_patterns_wordlist()]
    elif strategy == "mask":
        if not mask:
            raise ValueError("--mask 인자가 필요합니다 (예: ?d?d?d?d?d?d)")
        candidate_iters = [generate_from_mask(mask)]
    elif strategy == "hybrid":
        candidate_iters = [top_patterns_wordlist()]
        if mask:
            candidate_iters.append(generate_from_mask(mask))
    else:
        raise ValueError("strategy는 dict/mask/hybrid 중 하나여야 합니다.")

    # 공유 객체 생성
    counter = mp.Value('Q', 0)  # unsigned long long
    manager = mp.Manager()
    found_event = manager.Event()

    last_report = time.time()
    last_count = 0

    try:
        with mp.Pool(processes=processes, initializer=init_worker, initargs=(counter, found_event)) as pool:
            for it in candidate_iters:
                if found_event.is_set():
                    break
                tasks = ((zip_path, batch) for batch in chunker(it, batch_size))
                for result in pool.imap_unordered(_test_password, tasks, chunksize=1):
                    now = time.time()
                    if now - last_report >= progress_interval:
                        tried = counter.value
                        dt = now - t0
                        speed = (tried - last_count) / (now - last_report) if now - last_report > 0 else 0.0
                        overall = tried / dt if dt > 0 else 0.0
                        print(f"[{now_str()}] 시도={human_int(tried)} "
                              f"최근속도≈{speed:,.0f}/s 전체속도≈{overall:,.0f}/s 경과={dt:,.1f}s")
                        last_report = now
                        last_count = tried
                    if result:
                        pwd = result
                        found_event.set()
                        with open("password.txt", "w", encoding="utf-8") as f:
                            f.write(pwd + "\n")
                        dt = time.time() - t0
                        print(f"[{now_str()}] 성공! 비밀번호='{pwd}', 총 시도={human_int(counter.value)}, 경과={dt:,.1f}s")
                        return pwd
                if found_event.is_set():
                    break
    except KeyboardInterrupt:
        print("\n[!] 사용자가 중단했습니다.")
        return None

    dt = time.time() - t0
    print(f"[{now_str()}] 종료 :: 실패(미발견). 총 시도={human_int(counter.value)}, 경과={dt:,.1f}s")
    return None

def main():
    parser = argparse.ArgumentParser(description="6자리 소문자/숫자 ZIP 비밀번호 크래커")
    parser.add_argument("zip_path", help="대상 ZIP 파일 경로 (예: emergency_storage_key.zip)")
    parser.add_argument("--strategy", choices=["dict", "mask", "hybrid"], default="hybrid")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--progress-interval", type=float, default=2.0)
    args = parser.parse_args()

    if not os.path.exists(args.zip_path):
        print(f"파일을 찾을 수 없습니다: {args.zip_path}")
        sys.exit(1)

    if args.strategy in ("mask", "hybrid") and args.mask:
        if args.mask == "?a?a?a?a?a?a":
            print("[경고] 전체 키스페이스(36^6)를 직접 탐색하려 합니다.")

    pwd = unlock_zip(
        zip_path=args.zip_path,
        strategy=args.strategy,
        mask=args.mask,
        processes=args.processes,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
    )
    sys.exit(0 if pwd else 2)

if __name__ == "__main__":
    main()
