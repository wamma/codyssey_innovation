# -*- coding: utf-8 -*-
"""
door_hacking.py
---------------
Zip(ZipCrypto 또는 AES)으로 암호화된 파일의 6자리 소문자/숫자 비밀번호를
사전/마스크/병렬 브루트포스로 시도하는 스크립트.

사용 예시:
    python door_hacking.py emergency_storage_key.zip --strategy hybrid
    python door_hacking.py emergency_storage_key.zip --strategy mask --mask "?d?d?d?d?d?d"
    python door_hacking.py emergency_storage_key.zip --strategy dict
    python door_hacking.py emergency_storage_key.zip --processes 8

주의:
- 전체 키스페이스(36^6)는 매우 큽니다. 기본 설정은 "가능성 높은" 패턴부터 시도합니다.
- Zip이 AES로 암호화된 경우 표준 zipfile 모듈 대신 pyzipper가 자동으로 사용됩니다(설치되어 있을 경우).
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


LOWER = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
DIGIT = string.digits           # '0123456789'
ALNUM = LOWER + DIGIT           # 36 chars


def human_int(n: int) -> str:
    """정수 가독성 향상."""
    return f"{n:,}"


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _open_zip(zip_path: str):
    """Zip 파일 핸들 오픈 (AES/ZipCrypto 모두 지원)."""
    if _USE_PYZIPPER:
        return zfmod.AESZipFile(zip_path, 'r')
    return zfmod.ZipFile(zip_path, 'r')


def _test_password(args) -> Optional[str]:
    """
    워커 프로세스에서 실행: 비밀번호 후보 리스트를 받아 순차 테스트.
    found_event: 다른 프로세스가 찾으면 중단.
    counter: 시도 수를 배치로 더함.
    """
    zip_path, candidates, found_event, counter, batch_incr = args
    if found_event.is_set():
        return None

    try:
        with _open_zip(zip_path) as zf:
            names = zf.namelist()
            if not names:
                return None
            target = names[0]
            for pwd in candidates:
                if found_event.is_set():
                    return None
                try:
                    # pyzipper/zipfile 공통: pwd는 bytes 필요
                    with zf.open(target, 'r', pwd=pwd.encode('utf-8')) as f:
                        # 아주 적은 바이트만 읽어도 검증됨
                        _ = f.read(1)
                    # 성공
                    found_event.set()
                    return pwd
                except Exception:
                    pass
                finally:
                    # 배치 카운트 증가
                    with counter.get_lock():
                        counter.value += batch_incr
    except Exception:
        # zip 파일 자체 오류 등
        return None
    return None


def chunker(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    """후보를 size 단위로 끊어서 전달."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def generate_from_mask(mask: str) -> Iterable[str]:
    """
    마스크 문자열을 해석하여 후보 생성.
    토큰:
        ?d -> digit
        ?l -> lower
        ?a -> lower+digit
    예: "?l?l?d?d?d?d"
    """
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
            # 리터럴 문자도 지원 가능하지만, 여기선 제한
            raise ValueError("마스크는 '?d', '?l', '?a' 토큰만 지원합니다.")
    # 카테시안 곱
    for tup in itertools.product(*pools):
        yield ''.join(tup)


def top_patterns_wordlist() -> Iterable[str]:
    """
    '가능성 높은' 6자리 패턴을 우선적으로 생성하는 사전.
    - 숫자 6자리(생년, 날짜, 반복수 등)
    - 영문 6자리(간단 단어)
    - 혼합 대표 마스크 일부(?l?l?d?d?d?d, ?d?d?l?l?d?d 등)
    - common 프리픽스/서픽스 결합
    """
    # 1) 숫자 6자리: 000000 ~ 999999
    for n in range(0, 1_000_000):
        yield f"{n:06d}"

    # 2) 영문 6자리: 빈도 상위 알파벳 12자 위주(12^6 ≈ 2,985,984)
    freq_order = "etaoinshrdlucmfwypvbgkjqxz"
    top = freq_order[:12]
    for tup in itertools.product(top, repeat=6):
        yield ''.join(tup)

    # 3) 대표 혼합 마스크 일부
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

    # 4) 자주 쓰이는 프리/서픽스 결합(간단 예시)
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
    """
    zip_path의 비밀번호를 탐색.
    strategy:
        - "dict": top_patterns_wordlist()만
        - "mask": mask 인자 필수(예: "?d?d?d?d?d?d")
        - "hybrid": dict → mask(기본은 전역 마스크 '?a' * 6은 위험하므로 사용자가 지정한 경우만)
    찾으면 password.txt에 저장하고 반환. 실패 시 None.
    """

    if processes is None or processes <= 0:
        processes = max(1, mp.cpu_count() - 1)

    print(f"[{now_str()}] 시작 :: 파일='{zip_path}', 전략='{strategy}', 프로세스={processes}")
    t0 = time.time()

    # 후보 이터레이터 구성
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

    manager = mp.Manager()
    found_event = manager.Event()
    counter = manager.Value('Q', 0)  # unsigned long long
    last_report = time.time()
    last_count = 0

    try:
        with mp.Pool(processes=processes) as pool:
            for it in candidate_iters:
                if found_event.is_set():
                    break

                tasks = (
                    (zip_path, batch, found_event, counter, len(batch))
                    for batch in chunker(it, batch_size)
                )

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
    parser.add_argument("--strategy", choices=["dict", "mask", "hybrid"], default="hybrid",
                        help="공격 전략 선택 (기본: hybrid)")
    parser.add_argument("--mask", default=None,
                        help="마스크 문자열 (예: ?d?d?d?d?d?d, ?l?l?d?d?d?d)")
    parser.add_argument("--processes", type=int, default=None,
                        help="사용할 프로세스 수 (기본: CPU-1)")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="워커당 후보 배치 크기 (기본: 2000)")
    parser.add_argument("--progress-interval", type=float, default=2.0,
                        help="진행 로그 출력 간격(초) (기본: 2.0)")
    args = parser.parse_args()

    if not os.path.exists(args.zip_path):
        print(f"파일을 찾을 수 없습니다: {args.zip_path}")
        sys.exit(1)

    if args.strategy in ("mask", "hybrid") and args.mask:
        if args.mask == "?a?a?a?a?a?a":
            print("[경고] 전체 키스페이스(36^6)를 직접 탐색하려 합니다. 매우 오래 걸릴 수 있습니다.")

    pwd = unlock_zip(
        zip_path=args.zip_path,
        strategy=args.strategy,
        mask=args.mask,
        processes=args.processes,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
    )
    if pwd is None:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
