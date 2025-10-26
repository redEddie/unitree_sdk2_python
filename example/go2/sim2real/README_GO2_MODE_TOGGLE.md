# Unitree Go2 로봇 모드 전환 가이드

## 개요

Unitree Go2 로봇은 두 가지 주요 제어 모드를 가지고 있습니다:
- **MCF 모드**: 리모컨으로 제어 가능한 기본 모드
- **Low-level 제어 모드**: 스크립트로 직접 모터를 제어하는 모드

## 모드 구조

### MCF 모드 (기본 모드)
```python
CheckMode() 결과: {'form': '0', 'name': 'mcf'}
```
- **상태**: 리모컨 제어 가능 ✓
- **용도**: 일반적인 로봇 사용, 이동, 기본 동작
- **제약**: Low-level 스크립트 제어 불가

### Low-level 제어 모드
```python
CheckMode() 결과: {'form': '0', 'name': ''}
```
- **상태**: 스크립트로 모터 직접 제어 가능 ✓
- **용도**: 실험, 개발, 커스텀 모션 제어
- **제약**: 리모컨 제어 불가

## 사용하는 클래스 및 메소드

### MotionSwitcherClient

```python
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

# 초기화
msc = MotionSwitcherClient()
msc.SetTimeout(5.0)
msc.Init()
```

### 주요 메소드

#### CheckMode()
현재 활성화된 모드를 확인합니다.

```python
status, result = msc.CheckMode()
```

**반환값:**
- `status = 0`: 성공
- `result['name'] = 'mcf'`: MCF 모드 활성
- `result['name'] = ''`: Low-level 모드 활성 (모드 없음)

#### ReleaseMode()
현재 활성화된 모드를 해제합니다.

```python
code, _ = msc.ReleaseMode()
```

**사용:** MCF 모드 → Low-level 모드로 전환

#### SelectMode(name)
특정 모드를 선택하여 활성화합니다.

```python
code, _ = msc.SelectMode("mcf")
```

**사용:** Low-level 모드 → MCF 모드로 전환

## 모드 전환 로직

### MCF 모드 → Low-level 모드

```python
status, result = msc.CheckMode()

if result['name'] == 'mcf':
    code, _ = msc.ReleaseMode()
    if code == 0:
        print("Low-level 모드로 전환 완료")
```

### Low-level 모드 → MCF 모드

```python
status, result = msc.CheckMode()

if not result['name']:
    code, _ = msc.SelectMode("mcf")
    if code == 0:
        print("MCF 모드로 전환 완료")
```

## 사용 방법

```bash
# 스크립트 실행
python3 go2_mode_toggle.py eno1

# 옵션 선택
1. 모드 토글 (Low-level ↔ MCF)
2. 현재 모드만 확인
```

### 실험 워크플로우

1. **실험 시작 전**: MCF 모드 → Low-level 모드로 전환
2. **실험 진행**: Low-level 스크립트 실행 (예: `go2_stand_example.py`)
3. **실험 완료 후**: Low-level 모드 → MCF 모드로 전환
4. **리모컨 제어**: 다시 리모컨으로 로봇 조작 가능

## 실행 예시

```bash
# MCF 모드에서 시작
$ python3 go2_mode_toggle.py eno1
선택: 1
현재 모드: mcf
✓ 전환 완료! Low-level 제어 모드

# Low-level 실험 진행
$ python3 go2_stand_example.py eno1

# 실험 완료 후 MCF 모드로 복귀
$ python3 go2_mode_toggle.py eno1
선택: 1
현재 모드: Low-level 제어 모드
✓ 전환 완료! mcf (MCF 모드)
```

## 에러 코드

### 3104 에러
```
⚠ 오류: 모드 확인 실패 (status code: 3104)
```

**의미**: 모드 확인 통신 실패

**해결**: 잠시 시간이 지난 후 다시 실행하세요. 이는 로봇의 안전 메커니즘일 수 있으므로 강제로 우회하지 마세요.

### 7004 에러
```
⚠ 오류: 모드 선택 실패 (code: 7004)
```

**의미**: 모드 전환 실패

**해결**: 현재 상태에서 해당 모드로 전환할 수 없는 경우입니다.

## 주의사항

1. **모드 전환 전 로봇 상태 확인**
   - 로봇이 안전한 위치에 있는지 확인
   - 주변에 장애물이 없는지 확인

2. **Low-level 제어 중 리모컨 사용 불가**
   - Low-level 모드에서는 리모컨 명령이 무시됨
   - 실험 완료 후 반드시 MCF 모드로 복귀 필요

3. **에러 발생 시 우회하지 말 것**
   - 에러는 로봇의 안전 메커니즘일 수 있음
   - 시간을 두고 재시도하는 것을 권장

---

**작성일**: 2025-10-26  
**SDK**: unitree_sdk2_python  
**테스트 로봇**: Unitree Go2
