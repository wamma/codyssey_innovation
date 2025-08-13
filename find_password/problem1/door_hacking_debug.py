#!/usr/bin/env python3
"""
디버깅용 ZIP 비밀번호 크래커
더 다양한 문자셋과 패턴을 시도합니다.
"""

import argparse
import itertools
import string
import time
import zipfile
from typing import Generator, Optional


def test_password(zf: zipfile.ZipFile, member: str, candidate: str) -> bool:
    """비밀번호 후보를 시도해서 맞는지 확인"""
    try:
        with zf.open(member, pwd=candidate.encode('utf-8')) as f:
            content = f.read()
        print(f"🎉 비밀번호 찾음: '{candidate}'")
        print(f"내용: {content.decode('utf-8', errors='ignore')}")
        return True
    except Exception as e:
        return False


def comprehensive_search(zip_path: str) -> Optional[str]:
    """포괄적인 비밀번호 검색"""
    
    if not zipfile.is_zipfile(zip_path):
        print(f"오류: {zip_path}는 유효한 ZIP 파일이 아닙니다.")
        return None
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        if not members:
            print("오류: ZIP 파일이 비어있습니다.")
            return None
        
        member = members[0]  # 첫 번째 파일 사용
        print(f"테스트 대상: {member}")
        
        # 다양한 문자셋 정의
        charsets = {
            'digits': string.digits,  # 0-9
            'lowercase': string.ascii_lowercase,  # a-z
            'uppercase': string.ascii_uppercase,  # A-Z
            'letters': string.ascii_letters,  # a-zA-Z
            'alphanumeric': string.digits + string.ascii_letters,  # 0-9a-zA-Z
            'basic_special': string.digits + string.ascii_letters + '!@#$%^&*()',
            'all_printable': string.printable.replace(' \t\n\r\x0b\x0c', '')  # 공백문자 제외
        }
        
        # 다양한 길이 시도
        lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        start_time = time.time()
        total_attempts = 0
        
        # 각 문자셋과 길이 조합 시도
        for charset_name, charset in charsets.items():
            for length in lengths:
                print(f"\n[{charset_name}] {length}자리 패스워드 검색 중...")
                print(f"문자셋: '{charset[:20]}{'...' if len(charset) > 20 else ''}'")
                
                # 검색 공간 크기 계산
                search_space = len(charset) ** length
                print(f"검색 공간: {search_space:,}")
                
                # 너무 큰 검색 공간은 샘플링
                max_attempts = min(search_space, 100000)
                
                attempts_this_round = 0
                for candidate_tuple in itertools.product(charset, repeat=length):
                    candidate = ''.join(candidate_tuple)
                    total_attempts += 1
                    attempts_this_round += 1
                    
                    if test_password(zf, member, candidate):
                        elapsed = time.time() - start_time
                        print(f"\n총 시도 횟수: {total_attempts:,}")
                        print(f"소요 시간: {elapsed:.2f}초")
                        
                        # 결과 저장
                        with open('found_password.txt', 'w') as f:
                            f.write(candidate)
                        
                        return candidate
                    
                    # 진행상황 출력
                    if attempts_this_round % 1000 == 0:
                        print(f"  진행: {attempts_this_round:,}/{max_attempts:,} (현재: '{candidate}')")
                    
                    # 최대 시도 횟수 제한
                    if attempts_this_round >= max_attempts:
                        print(f"  최대 시도 횟수 {max_attempts:,}에 도달")
                        break
        
        print(f"\n총 {total_attempts:,}번 시도했지만 비밀번호를 찾지 못했습니다.")
        return None


def quick_common_passwords(zip_path: str) -> Optional[str]:
    """일반적인 비밀번호들을 빠르게 시도"""
    
    common_passwords = [
        # 숫자 패턴
        '123456', '000000', '111111', '123123', '654321',
        '012345', '543210', '999999', '888888', '777777',
        
        # 문자 패턴  
        'password', 'admin', 'qwerty', 'abc123', 'test',
        'secret', 'unlock', 'open', 'key', 'door',
        
        # 특수 패턴
        'emergency', 'storage', 'backup', 'temp', 'default',
        'guest', 'user', 'root', 'system', 'access',
        
        # 혼합 패턴
        'pass123', 'admin123', 'test123', 'key123',
        'door123', 'open123', 'temp123', 'user123'
    ]
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        member = zf.namelist()[0]
        print(f"일반적인 비밀번호 {len(common_passwords)}개 시도 중...")
        
        for i, password in enumerate(common_passwords):
            if test_password(zf, member, password):
                print(f"일반적인 비밀번호에서 찾음: '{password}'")
                return password
            
            if i % 10 == 0:
                print(f"  진행: {i}/{len(common_passwords)} (현재: '{password}')")
    
    print("일반적인 비밀번호에서 찾지 못함")
    return None


def main():
    parser = argparse.ArgumentParser(description='디버깅용 ZIP 비밀번호 크래커')
    parser.add_argument('zip_file', help='대상 ZIP 파일')
    parser.add_argument('--mode', choices=['quick', 'comprehensive'], 
                       default='quick', help='검색 모드')
    
    args = parser.parse_args()
    
    print(f"ZIP 파일: {args.zip_file}")
    print(f"모드: {args.mode}")
    
    if args.mode == 'quick':
        result = quick_common_passwords(args.zip_file)
    else:
        result = comprehensive_search(args.zip_file)
    
    if result:
        print(f"\n✅ 비밀번호를 찾았습니다: '{result}'")
    else:
        print(f"\n❌ 비밀번호를 찾지 못했습니다.")


if __name__ == '__main__':
    main() 