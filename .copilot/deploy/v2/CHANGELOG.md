# v2 Changelog (from v1)

> Last updated: 2026-05-04

## Critical Changes

### 1. Dual-Mode ActionScaler — Actor Input & Combine

학습 시 wrapper에 따라 ActionScaler 처리가 다름. v2는 두 모드 모두 지원:

**Mode A (`as_mode="wrapper"`)** — Lift, Stack, Drawer:
- Per-task wrapper가 ActionScaler를 내장
- Actor input: base_action의 pos+grip을 ActionScaler로 normalize, euler=0
- Combine: 정규화 공간에서 pos+grip 합산 → unscale back
- Rotation: 항상 physical delta (residual * rot_scale)

**Mode B (`as_mode="script_only"`)** — Cup, PnP:
- Wrapper에 ActionScaler 없음 (training script에서만 replay buffer용으로 사용)
- Actor input: raw base_action 7D
- Combine: 모두 physical delta (pos*pos_scale, rot*rot_scale, grip*grip_scale)

**모드 자동 선택**: TASKS dict의 `as_mode` 필드 + checkpoint의 `use_action_scaler` 조합.

### 2. TASKS dict — 정확한 dim/scale 반영

| Task | State | Object | Total+BA | grip_scale | grip_range | Special |
|------|-------|--------|----------|------------|------------|---------|
| Lift | 10 | 7 | 24 | 0.1 | [0,1] | — |
| PnP | 10 | 10 | 27 | 0.1 | [0,1] | — |
| Stack | 10 | 10 | 27 | 0.004 | [0,0.04] | — |
| Cup | 8 | 8 | 23 | 0.004 | [0,0.04] | gripper latch |
| Drawer | 8 | 5 | 20 | 0.004 | [0,0.04] | grip always 0 |

### 3. build_rl_state() — Task별 정확한 state 구성

- Lift: state(10) = pos3+quat4+grip_qpos2+contact1, object(7) = cube pos3+quat_wxyz4
- PnP: state(10) + object(10) = cube pos3+quat4 + plate pos3
- Stack: state(10) + object(10) = white_cube pos3+quat4 + green_pos3
- Cup: state(8) = pos3+quat4+grip1, object(8) = cup state
- Drawer: state(8) = pos3+quat4+grip1, object(5) = slide1+idx1+face_center3

### 4. Drawer — Grip Always Zero

Drawer is push task. Combined grip output is always forced to 0.0,
matching training wrapper behavior.

### 5. Cup — Gripper Latch Logic

Cup uses special gripper latch matching training wrapper:
- Close thresh: 0.015 (below → latch)
- Open thresh: 0.035 (above → count towards unlatch)
- Open steps: 32 consecutive open commands to unlatch
- Latch state resets per trial

### 6. ResidualActor.load_from_checkpoint

Auto-detects action_scale, use_action_scaler from checkpoint args.
Infers input/output dims from weight shapes.

## File Structure

```
.copilot/deploy/
├── v1/main_groot_residual.py     # original backup (2652 lines)
├── v2/main_groot_residual.py     # this version (3019 lines)
├── v2/CHANGELOG.md               # this file
├── action_scaler_stats.json      # min/max per task
└── BEST_CHECKPOINTS.md           # checkpoint paths & configs
```
