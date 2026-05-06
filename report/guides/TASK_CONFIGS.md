# Task별 하이퍼파라미터 설정

> 마지막 업데이트: 2026-05-02

## 하이퍼파라미터 비교표

| 항목 | Drawer (D2) | Stack Cube | PnP | Cube Lift |
|------|-------------|------------|-----|-----------|
| **train script** | `train_..._drawer.py` | `train_..._stack.py` | `train_..._pnp.py` | `train_..._mujoco.py` (원본) |
| **config file** | `config/residual_td3_mujoco_drawer.py` | (inline argparse) | `config/residual_td3_mujoco_pnp.py` | `config/residual_td3_mujoco.py` |
| **wrapper** | `mujoco_residual_wrapper_drawer.py` | `mujoco_residual_wrapper_stack.py` | `mujoco_residual_wrapper_pnp.py` | `mujoco_residual_wrapper.py` (원본) + `_cup.py` |
| **vec_env** | `mujoco_vec_env_drawer.py` | `mujoco_vec_env_stack.py` | (PnP 전용) | `mujoco_vec_env.py` (원본) |
| **eval script** | `eval_async_mujoco_drawer.py` | `eval_async_mujoco_stack.py` | `eval_async_mujoco_pnp.py` | `eval_async_mujoco.py` (원본) |

### 네트워크

| 항목 | Drawer | Stack | PnP | Lift |
|------|--------|-------|-----|------|
| actor_hidden_dim | 512 | **256** | 512 | 512 |
| critic_hidden_dim | 1024 | **256** | 1024 | 1024 |
| state_only | ✅ | ✅ (기본) | ❌ (이미지 사용) | ❌ |
| asymmetric_critic | ❌ | ❌ (argparse 옵션) | ✅ (depth) | ✅ |
| rl_cameras | [] | [] | front+wrist (depth) | front+wrist |

### 학습 알고리즘

| 항목 | Drawer | Stack | PnP | Lift |
|------|--------|-------|-----|------|
| actor_lr | **1e-5** | **3e-4** | 1e-5 | 1e-5 |
| critic_lr | 1e-4 | **3e-4** | 1e-4 | 1e-4 |
| action_scale | 0.1 (sweep: 0.03~0.1) | 0.1 | 0.1 | 0.1 |
| action_l2_reg | 10.0 (argparse) / **0.0 (SLURM default)** | **1.0** | 10.0 | 10.0 |
| gamma | 0.99 | **0.95** | 0.99 | 0.99 |
| n_step | 3 | 3 | 3 | 3 |
| batch_size | 256 | 256 | 256 | 256 |
| buffer_size | 200,000 | **500,000** | 200,000 | 200,000 |
| offline_fraction | 0.3 | 0.75 | 0.5 | 0.5 |
| critic_warmup_steps | 1,000 | **2,000** | 1,000 | 1,000 |
| target_tau | 0.005 | 0.005 | 0.005 | 0.005 |
| stddev_max | 0.05 | 0.05 | 0.05 | 0.05 |
| stddev_min | 0.05 | 0.05 | 0.05 | 0.05 |
| total_timesteps | 50,000 | 500,000 | 50,000 | 50,000 |
| update_every_n_steps | 1 | 1 | 1 | 1 |

### 환경

| 항목 | Drawer | Stack | PnP | Lift |
|------|--------|-------|-----|------|
| num_envs | 1 (argparse) / sweep에서 조정 | 1 (argparse) / **30 (SLURM)** | 1 | 1 |
| max_episode_steps | **500** | **500** | 300 | 300 |
| reward_type | **dense** (argparse default) / delta (sweep) | **dense** | dense_clipped | dense_clipped |
| success metric | drawer_closed | cube stacked | cube on plate (<3cm) | cube lifted |

> ⚠️ **argparse default vs SLURM preset**: 코드의 argparse default와 SLURM 스크립트의 환경변수 default가 다를 수 있음. SLURM 스크립트가 실제 실행 시 우선.

### Grip 처리

| Task | 방식 | 값 범위 | 비고 |
|------|------|---------|------|
| Drawer | **항상 0.0 고정** | N/A | 서랍 닫기라 grip 불필요 |
| Stack | **raw meters** | [0, 0.04] | MuJoCo gripper width |
| PnP | **normalized** | [0, 1] | 정규화된 grip 값 |
| Lift | **normalized** | [0, 1] | PnP와 동일 |

### ActionScaler

| Task | 사용 여부 | no_clamp | 비고 |
|------|----------|----------|------|
| Drawer | ✅ | ❌ (argparse에 없음, 수동 설정 필요) | offline data로 범위 계산 |
| Stack | ✅ | ✅ | 동일 |
| PnP | ❌ (미적용) | - | 물리 scale 방식 사용 |
| Lift | ✅ (코드에 있음, 실사용은 실험에 따라) | - | `mujoco_residual_wrapper.py`에 ActionScaler 코드 있음 |

## GR00T 체크포인트 경로

| Task | 경로 |
|------|------|
| Drawer | `~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000` |
| Stack | `~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000` |
| PnP | `~/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-100000` |
| Lift | `~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000` |

## Offline Data 경로

| Task | 경로 | 비고 |
|------|------|------|
| Drawer | `outputs/offline_drawer_zgate_100k` | z_gate 적용된 100K transitions |
| Stack | `outputs/offline_stack_66ep` | 66 episode base policy rollout |
| PnP | `outputs/offline_data/pnp_${DIFFICULTY}` | difficulty별 분리 |
| Lift | (미확인) | |

## GR00T Chunking 설정

모든 task 공통:
- `action_horizon = 16`: GR00T가 한번에 16 step action 예측
- `open_loop_horizon = 16`: 16 step 동안 캐싱된 action 재사용
- **eval 시 반드시 동일하게 설정** (horizon=1로 하면 base policy 성능 붕괴)
