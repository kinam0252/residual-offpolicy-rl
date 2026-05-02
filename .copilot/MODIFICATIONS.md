# 코드 수정사항 (원본 대비)

> 마지막 업데이트: 2026-05-02

## 요약

공유 코드 (모든 task 영향)와 task-specific 코드를 구분합니다.

## 🌐 공유 코드 수정 (Committed)

### 1. ActionScaler 기능 추가 — `normalization.py`
- **커밋**: `1faf05b` feat: add ActionScaler mode for data-driven residual scaling
- **파일**: `resfit/rl_finetuning/utils/normalization.py`
- **설명**: 하드코딩된 물리 스케일 대신 offline data 기반으로 action 범위를 자동 계산
- **영향**: ActionScaler를 사용하는 모든 task

### 2. ActionScaler no_clamp 옵션 — `normalization.py`
- **커밋**: `6475377` ActionScaler: add --no_action_clamp to preserve base policy
- **파일**: `resfit/rl_finetuning/utils/normalization.py`
- **설명**: `no_clamp=True`일 때 scale/unscale이 순수 선형 변환. base policy action이 offline 범위 밖이어도 잘리지 않음
- **영향**: ActionScaler를 사용하는 모든 task

### 3. q_agent state_only 분기 — `q_agent.py`
- **커밋**: `205d297` Stack Cube: offline data, ActionScaler smoke test, sweep scripts, base eval
- **파일**: `resfit/rl_finetuning/off_policy/rl/q_agent.py` (line ~940)
- **설명**: `state_only=True`일 때 state dim == 10 assert 스킵 (Drawer는 state dim이 다름)
- **영향**: state_only=True인 모든 task

### 4. Replay Buffer 공간 불일치 수정
- **커밋**: `740f7cb` fix: store normalized combined action in replay for ActionScaler mode
- **파일**: `mujoco_residual_wrapper.py` (원본 Lift wrapper) — 다른 task wrapper에도 같은 패턴 반영 필요
- **설명**: replay에 residual만 저장 → combined (base+residual) 저장으로 변경. critic/actor 공간 일치

### 5. Eval object_state_dim 수정
- **커밋**: `bb63bb5`
- **설명**: eval QAgent 생성 시 object_state_dim 10→7

### 6. ActionScaler attribute access 수정
- **커밋**: `785b9fc`

### 7. ActionScaler eval subprocess 전달
- **커밋**: `c938641`

## 📁 Task-Specific 수정

### Drawer (Uncommitted)

| 파일 | 변경 | 상태 |
|------|------|------|
| `mujoco_residual_wrapper_drawer.py` | **Replay clamp 버그 수정** — pre-clamp → post-clamp 값 저장 | ⚠️ uncommitted |
| `eval_async_mujoco_drawer.py` | `--eval_stddev` arg 추가 | ⚠️ uncommitted |
| `train_residual_td3_mujoco_drawer.py` | `--eval_stddev` arg + eval subprocess에 전달 | ⚠️ uncommitted |
| `slurm_train_drawer_td3_d2only.sh` | `EVAL_STDDEV` 환경변수 지원 | ⚠️ uncommitted |

### Stack (Uncommitted)

| 파일 | 변경 | 상태 |
|------|------|------|
| `mujoco_residual_wrapper_stack.py` | `render_mode` "rl_only" → "none" (불필요 렌더링 제거) | ⚠️ uncommitted |
| `mujoco_vec_env_stack.py` | render_mode 관련 수정 | ⚠️ uncommitted |
| `eval_async_mujoco_stack.py` | 수정사항 있음 | ⚠️ uncommitted |
| `train_residual_td3_mujoco_stack.py` | ActionScaler 관련 수정 | ⚠️ uncommitted |
| `slurm_sweep_stack_*.sh` | Stack sweep scripts (4개) | ⚠️ uncommitted |

### PnP / Lift
- **수정 없음** — 원본 코드 그대로

## 🆕 새로 만든 파일 (Untracked)

| 파일 | 용도 |
|------|------|
| `scripts/standalone_eval_drawer.py` | Drawer 독립 eval script (correct horizon=16) |
| `scripts/sbatch_oneshot_eval.sh` | One-shot eval SLURM script |
| `scripts/quick_eval_noise.py` | Quick eval noise test (삭제 가능) |
| `scripts/slurm_sweep_stack_l2_{1,2,3,4}.sh` | Stack L2 sweep scripts |

## Git 상태

```bash
# Committed (pushed까지 된 것)
git log origin/mujoco..HEAD  # local only commits 확인

# Uncommitted 확인
git status --short

# 전체 diff 확인
git diff --stat
```

## ⚠️ 주의: Cross-Task 영향

1. **공유 코드 (`q_agent.py`, `normalization.py`)** 수정은 모든 task에 영향
2. **각 task wrapper/train/eval**은 task별 독립 — 다른 task에 영향 없음
3. Drawer의 replay clamp 수정이 Stack에도 필요한지 **미확인** (같은 패턴 코드가 Stack wrapper line 421-425에 존재)
