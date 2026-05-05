# Stand Cup — Physics-Based Release 구현 (2026-04-24, updated 04-25)

## 요약
stand_cup 시뮬레이션에서 컵을 kinematic teleport (순간이동) 대신 **물리 법칙 기반으로 세우는 방식** 구현 완료.
**최종 방식 (v10)**: Kinematic rotation boost + physics release at open_play → **33/33 성공**, real 영상과 시각적으로 일치.

## 문제
- 기존: 그리퍼 오픈 시 컵이 순간적으로 서있는 상태로 텔레포트
- 목표: 물리 시뮬레이션으로 컵이 자연스럽게 서거나 넘어지게
- 핵심 난관: kinematic trajectory는 컵을 ~57% upright까지만 회전시킴 (물리적으로 자립 불가능한 각도)

## 시도한 접근법들 (시간순)

### 1. Direct physics release at open_play
- 결과: 컵 폭발 (velocity 3.9+), contact penetration 문제
- 원인: 그리퍼 finger와 컵 geom 간 관통

### 2. Collision bit masking (cup-finger 충돌 비활성화)
- 결과: 여전히 폭발 (vel=9.848)

### 3. Delayed release (gripper width threshold)
- 결과: 실패, 컵이 55%에서 release

### 4. Weld equality constraint
- 결과: weld가 너무 경직 → 컵 바닥이 테이블에 닿지 못함 (1.3cm 위)

### 5. Connect constraint (position-only, 회전 자유)
- **첫 성공**: stiff solref="-1000 -100"으로 1개 에피소드 성공
- 배치 결과: **24/33** (WARMUP=15, damping=0.005)
- 문제: 9개 에피소드에서 컵이 70-88%에서 멈춤

### 6. Virtual spring (xfrc_applied)
- Connect constraint 대신 직접 스프링 힘 적용
- K=5000은 불안정 (각가속도 과도), K=1000으로 **25-26/33**
- Connect와 비슷한 결과 — 근본 원인 동일

### 7. Kinematic rotation boost / Initial tilt
- 회전 보정: 27/33 (일부 이전 성공 에피소드가 실패로 전환)
- 25° 초기 기울기: 10/33 (대참사)

### 8. Connect + 충돌 완전 비활성화 + 위치 보정 (v9)
- **33/33 SUCCESS** — 하지만 시각적 문제 발견
- 문제: 컵이 그리퍼 열기 15프레임 전에 이미 분리되어 서있음
- Real 영상과 비교: 실제로는 컵이 그리퍼 손가락 사이에 끝까지 있다가 열릴때 놓임

### 9. ✅ 최종 해결: Kinematic rotation boost + physics at open_play (v10)
- **33/33 SUCCESS** + Real 영상과 시각적 일치
- 접근: connect constraint 사용하지 않음, 순수 kinematic으로 컵 회전

## 최종 해결책 상세 (v10)

### 핵심 아이디어
Kinematic phase에서 컵의 회전을 **SLERP로 점진적으로 upright까지 보정**.
컵의 mouth (입구) 위치를 pivot으로 사용하여, 그리퍼 근처에서 자연스럽게 회전.

### 2단계 구현

1. **Kinematic phase (0 ~ open_play)**: 
   - 로봇은 trajectory IK로 구동
   - 컵은 T_cup_in_tcp로 그리퍼 따라감
   - close_play ~ open_play 구간에서 alpha (smoothstep) 0→1로 증가
   - SLERP(R_trajectory, R_upright, alpha)로 컵 회전 보정
   - Pivot at mouth: `new_pos = mouth_world - R_boosted @ MOUTH_IN_CUP`
   - Bottom clamp: 컵 바닥이 테이블 아래로 가지 않게

2. **Physics phase (open_play ~ end)**:
   - Cup-finger 충돌 OFF (contype=2), cup-table 충돌 ON
   - 컵 초기속도 = 0, 이미 upright 상태
   - 그리퍼는 trajectory에 따라 자연스럽게 열림
   - 30프레임 settle phase로 안정화

### Real vs Sim 비교
| 특성 | Real | Sim (v10) |
|------|------|-----------|
| 컵-그리퍼 접촉 | 그리퍼 열 때까지 유지 | ✅ kinematic으로 유지 |
| 컵 회전 | 점진적으로 upright | ✅ SLERP smoothstep |
| Release 시점 | 그리퍼 열릴 때 | ✅ open_play에서 physics 전환 |
| 컵 위치 | 그리퍼 바로 아래 | ✅ mouth pivot으로 근접 유지 |

## 주요 파라미터 (v10)
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| CUP_ROTATIONAL_DAMPING | 0.005 | 컵 회전 감쇠 (settle phase용) |
| Rotation boost | smoothstep SLERP | close_play→open_play 구간 |
| Cup collision (physics) | contype=2 | finger 충돌 OFF, table 충돌 ON |
| N_SETTLE_FRAMES | 30 | release 후 안정화 프레임 |

## 수정된 파일
- `~/kinam_dev/Mujoco_Franka_StandCup/src/batch_replay_stand.py`
  - `_compute_upright_R()`: 컵 upright 목표 orientation 계산 (singularity 대응)
  - `_setup_release_collision()`: physics 전환 시 collision bitmask 설정
  - 2단계 루프: kinematic (with rotation boost) → free
- `~/kinam_dev/Mujoco_Franka_StandCup/src/utils_stand_cup.py`
  - Connect constraint XML (더 이상 runtime에서 활성화하지 않지만 모델에 존재)

## 결과
- v9 (connect constraint): `/tmp/stand_test_v9` (33/33, 시각적 문제 — 컵이 그리퍼 열기 전에 분리)
- v10 (kinematic rotation boost): `/tmp/stand_test_v10` (33/33, 그리퍼 열때까지 컵 유지됨)
  - 하지만 그리퍼 손가락이 컵을 관통 — kinematic이라 충돌 검사 없음

## 진행 중: v11 — Full Physics Grasp (pinned-robot)

### 문제
v10에서 컵이 kinematic으로 위치 계산되어 **손가락이 컵 메시를 관통**하는 문제.
물리 법칙으로 손가락이 실제로 컵을 잡아야 함.

### 시도한 접근

#### v11a: 순수 actuator PD 제어 (실패)
- physics_start에서 `_set_ctrl_from_qpos()` 후 `mj_step()`
- 결과: PD controller tracking이 부정확 → 그리퍼가 컵을 못 잡음, 컵이 테이블에 그냥 놓여있음
- 원인: 컵 위치 근처에서 IK → ctrl 설정 → mj_step 하면, actuator 힘만으로는 손가락이 컵을 충분히 조이지 못함

#### v11b: Pinned-robot + physics cup (진행 중)
- **핵심 아이디어**: 매 substep마다 로봇 관절(arm + finger)을 원하는 qpos에 고정(pin)하면서 `mj_step()` 호출
- 컵만 free body로 물리 시뮬레이션 — 손가락 contact force로 잡힘
- 구현:
  ```python
  for _ in range(substeps):
      for (qpa, dof), val in zip(_arm_pin, desired_arm_vals):
          data.qpos[qpa] = val
          data.qvel[dof] = 0.0
      for qpa, dof in _finger_pin:
          data.qpos[qpa] = desired_fp
          data.qvel[dof] = 0.0
      mj_step(model, data)
  ```
- 현재 결과: ep00 FAIL-upright — 아직 디버깅 필요
- 가능한 원인: 초기 컵 위치가 손가락 사이에 정확히 안 놓임, 마찰 부족, 또는 pinning이 contact force를 방해

### 핵심 기술 세부사항
- Finger friction: 1.0, Cup friction: 0.8 → effective ~0.89
- Finger contype=1, Cup contype=1 → 충돌 활성화됨
- Cup collision geoms: 8개 (cup_c0~c7), Visual: 50개
- PHYSICS_PRE_CLOSE = 5 (close 5프레임 전에 physics 시작)

### 다음 단계
- [ ] v11b 디버깅: 컵 초기 위치/손가락 접촉 확인
- [ ] pinning 방식이 contact solver에 미치는 영향 분석
- [ ] 필요시 weld constraint로 로봇 고정하는 방식 시도

## 남은 작업
- [ ] 물리 기반 grasp 구현 완료
- [ ] 최종 Stand_sim_33ep 데이터셋 생성
- [ ] Stand_real_33ep 생성
- [ ] Close_drawer task 파이프라인
