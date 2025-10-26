#!/usr/bin/env python3
"""
Go2 로봇 모드 토글 스크립트
Low-level 제어 모드 <-> MCF 모드 전환
"""
import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


def toggle_mode():
    """
    현재 모드 상태를 확인하고 토글
    - 모드 없음 (Low-level) → MCF 모드로 전환 (리모컨 제어)
    - 모드 있음 (MCF 등) → Low-level 모드로 전환 (스크립트 제어)
    """
    print("\n" + "="*60)
    print("  Go2 로봇 모드 토글")
    print("="*60)
    
    # MotionSwitcherClient 초기화
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    
    # 현재 모드 확인
    print("\n[1단계] 현재 모드 확인 중...")
    status, result = msc.CheckMode()
    
    if status != 0:
        print(f"⚠ 오류: 모드 확인 실패 (status code: {status})")
        return
    
    current_mode = result.get('name', None) if result else None
    
    if current_mode:
        # 모드가 있음 → Low-level 모드로 전환
        print(f"현재 모드: {current_mode}")
        print("\n[2단계] Low-level 제어 모드로 전환 중...")
        print("  → 모드 해제 (ReleaseMode)")
        
        code, _ = msc.ReleaseMode()
        
        if code == 0:
            time.sleep(0.5)
            # 확인
            status, result = msc.CheckMode()
            new_mode = result.get('name', None) if result else None
            
            print("\n" + "="*60)
            print("✓ 전환 완료!")
            print(f"  이전: {current_mode}")
            print(f"  현재: Low-level 제어 모드 (모드 없음)")
            print("  → 스크립트로 로봇 제어 가능")
            print("  → 리모컨 제어 불가")
            print("="*60)
        else:
            print(f"\n⚠ 오류: 모드 해제 실패 (code: {code})")
    
    else:
        # 모드가 없음 → MCF 모드로 전환
        print("현재 모드: Low-level 제어 모드 (모드 없음)")
        print("\n[2단계] MCF 모드로 전환 중...")
        print("  → 모드 선택 (SelectMode: 'mcf')")
        
        code, _ = msc.SelectMode("mcf")
        
        if code == 0:
            time.sleep(0.5)
            # 확인
            status, result = msc.CheckMode()
            new_mode = result.get('name', None) if result else None
            
            print("\n" + "="*60)
            print("✓ 전환 완료!")
            print(f"  이전: Low-level 제어 모드")
            print(f"  현재: {new_mode} (MCF 모드)")
            print("  → 리모컨 제어 가능")
            print("  → 스크립트 제어 불가")
            print("="*60)
        else:
            print(f"\n⚠ 오류: 모드 선택 실패 (code: {code})")


def check_mode_only():
    """
    현재 모드만 확인
    """
    print("\n" + "="*60)
    print("  현재 모드 확인")
    print("="*60)
    
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    
    status, result = msc.CheckMode()
    
    if status == 0 and result:
        mode_name = result.get('name', None)
        if mode_name:
            print(f"\n현재 모드: {mode_name}")
            print("상태: 리모컨 제어 가능")
        else:
            print(f"\n현재 모드: Low-level 제어 모드 (모드 없음)")
            print("상태: 스크립트 제어 가능, 리모컨 불가")
        
        print(f"\n전체 결과: {result}")
    else:
        print(f"⚠ 오류: status={status}, result={result}")
    
    print("="*60)


if __name__ == '__main__':
    # 네트워크 초기화
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    
    print("\n사용 가능한 명령:")
    print("1. 모드 토글 (Low-level ↔ MCF)")
    print("2. 현재 모드만 확인")
    
    choice = input("\n선택 (1 또는 2): ").strip()
    
    if choice == "1":
        try:
            toggle_mode()
        except KeyboardInterrupt:
            print("\n\n중단되었습니다.")
        except Exception as e:
            print(f"\n오류 발생: {e}")
    elif choice == "2":
        try:
            check_mode_only()
        except Exception as e:
            print(f"\n오류 발생: {e}")
    else:
        print("잘못된 선택입니다.")