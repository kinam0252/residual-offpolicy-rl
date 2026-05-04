# Real Robot Deployment: GR00T Base + B200 Residual RL

## Overview

Real robot에서 GR00T VLA (base policy)와 B200에서 학습된 Residual RL policy를
결합하여 rollout하는 시스템의 구조 설명.

```
┌─────────────────────────────────────────────────────────┐
│              Real Robot Inference Loop                   │
│                                                         │
│  Camera (base+wrist)  ──→  GR00T VLA  ──→  base_action  │
│                                             (pos, quat,  │
│                                              grip) ×16   │
│  Robot state (ee_pos,  ──→  Residual   ──→  residual_7d  │
│   ee_quat, grip, obj)      Actor (MLP)     (Δpos, Δrot,  │
│                                              Δgrip)      │
│                                                         │
│  combined = combine_actions(base, residual)             │
│  ──→ /target_pose (ROS2) ──→ Franka FR3                 │
└─────────────────────────────────────────────────────────┘
```

## 파일 위치

| 파일 | 역할 |
|------|------|
| `~/kinam_dev/execute_real_robot/main_groot_residual.py` | Real robot 배포 코드 (2652줄) |
| `~/kinam_dev/checkpoints/{task}/rl_best/` | B200에서 가져온 residual 체크포인트 |
| `B200:~/Repos/Intern/residual-offpolicy-rl/` | Residual RL 학습 코드 |

---

## 1. Base Policy (GR00T VLA)

GR00T는 vision-language-action 모델로 카메라 이미지 2장 + 현재 EEF 상태를 입력받아
**16-step absolute action chunk**을 출력.

```python
# L2091~2107: GR00T inference
result = inferencer.predict(
    cam_base=cam_base_resized,
    cam_wrist=cam_wrist_resized,
    eef_pos=real_ee_pos,
    eef_quat=real_ee_quat,
    gripper_width=gripper_width,
)
pred_pos  = result["eef_pos"]       # (16, 3) xyz meters
pred_quat = result["eef_quat"]      # (16, 4) xyzw quaternion
pred_grip = result["gripper_width"] # (16,)   normalized or raw meters
```

---

## 2. Residual Actor (B200에서 학습)

### 2.1 네트워크 구조

```python
# L472~492: ResidualActor
class ResidualActor(nn.Module):
    # 3-layer MLP: input → 256 → 256 → 7
    # LayerNorm + ReLU, Tanh output * action_scale
    def forward(self, state, base_action_7d):
        x = torch.cat([state, base_action_7d], dim=-1)
        return self.policy(x) * self.action_scale
```

**Input**: `state` (로봇+오브젝트) + `base_action_7d` (GR00T 출력 변환)
**Output**: `residual_7d` in `[-action_scale, action_scale]`

### 2.2 State 구성 (현재: PnP 20D)

```python
# L2131~2154: state 구성
state_10d = [ee_pos(3), ee_quat_xyzw(4), gripper_qpos(2), contact(1)]  # 로봇 10D
obj_state_10d = [cube_pos(3), cube_quat_wxyz(4), bowl_pos(3)]          # 오브젝트 10D
state_20d = concat(state_10d, obj_state_10d)                           # 총 20D
```

⚠️ **Task별로 state 차원이 다름** (아래 참고)

### 2.3 Base Action 변환

GR00T 출력 (pos + quat + grip) → Actor 입력용 7D 변환:

```python
# L544~551: build_base_action_7d
base_action_7d = [pos(3), euler_xyz(3), grip(1)]  # quat → euler 변환
```

### 2.4 체크포인트 로딩

```python
# L494~527: load_from_checkpoint
ckpt = torch.load(ckpt_path)
# args에서 action_scale 자동 읽기
# model dict에서 actor.policy.* 가중치 추출
# input_dim, hidden_dim, output_dim 자동 감지
```

---

## 3. Combine Actions (Legacy 모드)

현재 배포 코드는 **Legacy 물리 스케일 모드**만 지원:

```python
# L529~541: combine_actions
combined_pos  = base_pos + residual[:3] * pos_scale   # default 0.02m
combined_rot  = base_rot * Rotation.from_euler("xyz", residual[3:6] * rot_scale)  # 0.05rad
combined_grip = clip(base_grip + residual[6] * grip_scale, 0, 1)  # 0.1
```

### CLI 파라미터

```
--residual-checkpoint   체크포인트 경로 (None=비활성)
--residual-pos-scale    위치 스케일 (default: 0.02m)
--residual-rot-scale    회전 스케일 (default: 0.05rad)
--residual-grip-scale   그리퍼 스케일 (default: 0.1)
--residual-action-scale Actor output scale (default: 0.1)
```

---

## 4. ActionScaler 모드 (B200 최신 — 미지원, 마이그레이션 필요)

B200 최신 코드는 **ActionScaler**를 사용하여 data-driven normalization:

```python
# B200 wrapper: _combine_actions (ActionScaler mode)
base_pg = [base_pos(3), base_grip(1)]        # 4D 물리값
base_pg_norm = scaler.scale(base_pg)          # → [-1, 1] 정규화
combined_norm = base_pg_norm + residual_pg    # 정규화 공간에서 합산
combined_phys = scaler.unscale(combined_norm) # → 물리 공간 복원
```

**핵심 차이**: Actor에 들어가는 `base_action`도 정규화됨 (pos+grip → [-1,1], euler=0)

```python
# B200 wrapper: _augment_obs (ActionScaler mode)
pg = [base_pos(3), base_grip(1)]
pg_norm = scaler.scale(pg)              # 정규화
base_action_7d = [pg_norm[:3], zeros(3), pg_norm[3]]  # euler=0
# → Actor는 정규화된 base_action을 보고 정규화된 residual을 출력
```

### ActionScaler 통계 (task별)

`action_scaler_stats.json` 참고. offline demo 데이터의 pos+grip min/max에서 계산.

---

## 5. Task별 차이점

| Task | State dim | Obj state | Actor input | ActionScaler | grip_scale |
|------|-----------|-----------|-------------|-------------|------------|
| **Lift** | 10 (eef10) | 0 | 17 (10+7) | ❌ | 0.1 |
| **PnP** | 20 (eef10+obj10) | cube7+bowl3 | 27 (20+7) | ✅ (통합코드) | 0.1 |
| **Drawer** | 13 (eef8+obj5) | slide+idx+face3 | 20 (13+7) | ✅ | 0.004 |
| **Stack** | 22 (eef8+obj14) | TBD | 29 (22+7) | ✅ | 0.004 |
| **Cup** | 16 (eef8+obj8) | TBD | 23 (16+7) | ✅ | 0.004 |

**eef10**: pos(3)+quat(4)+gripper_qpos(2)+contact(1)
**eef8**: pos(3)+quat(4)+gripper_width(1)

---

## 6. 마이그레이션 TODO

`main_groot_residual.py`에 ActionScaler 지원 추가 필요:

1. **ActionScaler 클래스 포팅** — scale/unscale (task별 min/max)
2. **load_from_checkpoint 확장** — use_action_scaler 감지, scaler stats 로드
3. **combine_actions 듀얼 모드** — Legacy or ActionScaler
4. **build_base_action_7d 확장** — ActionScaler에서 정규화
5. **Task별 state builder** — drawer 13D, stack 22D 등
6. **체크포인트에 scaler stats 저장** — B200 코드 수정

자세한 계획: session plan.md 참고
