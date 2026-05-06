# Task 전환 체크리스트

> Task를 전환할 때 아래 항목을 순서대로 확인하세요.

## 1. 파일 경로 확인

- [ ] **Train script**: `resfit/rl_finetuning/scripts/train_residual_td3_mujoco_{task}.py`
  - ⚠️ Lift는 `train_residual_td3_mujoco.py` (접미사 없음)
- [ ] **Eval script**: `resfit/rl_finetuning/scripts/eval_async_mujoco_{task}.py`
  - ⚠️ Lift는 `eval_async_mujoco.py` (접미사 없음)
- [ ] **Wrapper**: `resfit/rl_finetuning/wrappers/mujoco_residual_wrapper_{task}.py`
  - ⚠️ Lift는 `mujoco_residual_wrapper.py` (원본) + `_cup.py`
- [ ] **Vec env**: `resfit/rl_finetuning/wrappers/mujoco_vec_env_{task}.py`
  - ⚠️ Lift는 `mujoco_vec_env.py` (접미사 없음)
- [ ] **Config**: `resfit/rl_finetuning/config/residual_td3_mujoco_{task}.py` (있으면)
- [ ] **SLURM script**: `scripts/slurm_train_{task}_td3*.sh`

## 2. 경로 설정

- [ ] **GR00T checkpoint 경로** — task별로 다름 (→ `TASK_CONFIGS.md` 참조)
- [ ] **Offline data 경로** — task별로 다름
- [ ] **Output directory** — 겹치지 않게 task명 포함

## 3. 하이퍼파라미터 확인

> Stack은 다른 task 대비 매우 다른 설정을 사용합니다!

- [ ] **actor_lr**: Drawer/PnP/Lift = 1e-5, Stack = 3e-4
- [ ] **critic_lr**: Drawer/PnP/Lift = 1e-4, Stack = 3e-4
- [ ] **action_l2_reg**: Drawer/PnP/Lift = 10.0, Stack = 1.0
- [ ] **gamma**: Drawer/PnP/Lift = 0.99, Stack = 0.95
- [ ] **buffer_size**: Drawer/PnP/Lift = 200K, Stack = 500K
- [ ] **offline_fraction**: Drawer = 0.3, PnP/Lift = 0.5, Stack = 0.75
- [ ] **hidden_dim**: Drawer/PnP/Lift = 512/1024, Stack = 256/256
- [ ] **num_envs**: 각 task의 slurm script에서 확인
- [ ] **total_timesteps**: 각 task의 목표 확인

## 4. Task-Specific 설정

### Grip 처리
- [ ] **Drawer**: grip = 항상 0.0 (고정, 서랍이라 grip 불필요)
- [ ] **Stack**: grip = raw meters [0, 0.04]
- [ ] **PnP / Lift**: grip = normalized [0, 1]

### Reward
- [ ] **Drawer**: `dense` (argparse default) / `delta` (sweep에서 사용)
- [ ] **Stack**: `dense` (argparse default)
- [ ] **PnP**: `dense_clipped`
- [ ] **Lift**: `dense_clipped`
- [ ] ⚠️ SLURM script에서 override하는 값과 argparse default가 다를 수 있음

### Observation 구조
- [ ] **Drawer**: state-only (카메라 없음), `state_only=True`
- [ ] **Stack**: state-only (기본), 옵션으로 `--asymmetric_critic`
- [ ] **PnP**: 이미지 사용 (front + wrist depth), `asymmetric_critic=True`
- [ ] **Lift**: 이미지 사용, `asymmetric_critic=True`

## 5. ActionScaler 확인

- [ ] `--use_action_scaler` 플래그 사용 여부
- [ ] `--no_action_clamp` 플래그 (Drawer argparse에는 없음 — Stack만 지원. Drawer는 코드 수정 필요)
- [ ] offline data에서 ActionScaler 통계가 올바르게 계산되는지

## 6. 알려진 버그 적용 확인

> → `KNOWN_ISSUES.md` 참조

- [ ] **Replay clamp 버그**: Drawer는 수정됨 (uncommitted). Stack은 **미확인** — 같은 패턴의 코드가 존재
- [ ] **open_loop_horizon**: eval에서 반드시 16으로 설정 (1이면 0% SR)
- [ ] **q_agent state_only 분기**: state_only=True task에서 state dim assert 우회 (committed)

## 7. Eval 설정

- [ ] `open_loop_horizon = 16` (GR00T chunking과 일치)
- [ ] `eval_stddev = 0.0` (deterministic eval, 기본값)
- [ ] eval episodes 수 (빠른 테스트: 10, 신뢰성 있는 평가: 50~100)
- [ ] standalone eval script 사용 가능: `scripts/standalone_eval_drawer.py`

## 8. SLURM 제출 전 확인

- [ ] 파티션 선택 (→ `SLURM_GUIDE.md` 참조)
- [ ] GPU 할당량 여유 확인 (`squeue -u kinamkim | wc -l`)
- [ ] `PYTHONUNBUFFERED=1` 환경변수 설정됨
- [ ] wandb mode 설정 (online vs offline)
- [ ] output/log 경로가 기존 실험과 겹치지 않는지
