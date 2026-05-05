# Best Checkpoints & Config Reference

> Last updated: 2026-05-04

⚠️ **ActionScaler 버전 체크포인트만 사용** (`use_action_scaler=True`).
경로 기준: `~/Repos/Intern/residual-offpolicy-rl/`

## Deploy Mode Reference

| Task | as_mode | Wrapper Used | Actor Input | Combine |
|------|---------|-------------|-------------|---------|
| Lift | `wrapper` | mujoco_residual_wrapper.py | AS normalize | AS combine |
| Stack | `wrapper` | _stack.py | AS normalize | AS combine |
| Drawer | `wrapper` | _drawer.py | AS normalize | AS combine |
| Cup | `script_only` | _cup.py (no AS) | raw | physical delta |
| PnP | `script_only` | unified | raw | physical delta |

---

## Cube Lift — **80% SR** (ActionScaler version)

| Key | Value |
|-----|-------|
| **Best checkpoint** | `outputs/mujoco_td3/gr00t_sim_66ep/as_sweep_hard_as01_noL2/checkpoints/best.pt` |
| **GR00T checkpoint** | `~/DATA/INTERN/training/gr00t_sim_66ep/checkpoint-300000` |
| **Offline data** | `~/Repos/Intern/outputs/offline_data/gr00t_sim_66ep/hard` (149,614 samples, 320 files) |
| **action_scale** | 0.1 |
| **action_l2_reg** | 0.0 |
| **Code version** | per-task (구버전, `mujoco_td3` sweep) |
| **ActionScaler stats key** | `lift` |

### Lift Top 3 (ActionScaler=True only)
| # | Dir | AS | L2 | Best SR | Last SR | W_max | Notes |
|---|-----|----|----|---------|---------|-------|-------|
| 🥇 | `as_sweep_hard_as01_noL2` | 0.1 | 0.0 | **80.0%** | **67.5%** | 0.254 | ✅ **BEST** 안정적 |
| 🥈 | `as_sweep_hard_as03_l2001` | 0.3 | 0.01 | 80.0% | 0.0% | 0.336 | ✅ **2nd** last 불안정 |
| 🥉 | `as_sweep_hard_as01_l2001` | 0.1 | 0.01 | 70.0% | 30.0% | 0.219 | ✅ **3rd** |

---

## Close Drawer — **100% SR** (from 50% base)

| Key | Value |
|-----|-------|
| **Best checkpoint** | `outputs/d2only_td3_as0.05_L20.01_cw1000_delta_H_20260502_095333/checkpoints/best.pt` |
| **GR00T checkpoint** | (D2-only, no GR00T) |
| **Offline data** | `outputs/offline_drawer_zgate_100k` (16,810 samples, 90 files) |
| **action_scale** | 0.05 |
| **action_l2_reg** | 0.01 |
| **Special** | delta reward, critic_warmup=1000 |
| **Code version** | per-task (D2-only script) |
| **ActionScaler stats key** | `drawer` |
| **Output W_max** | 0.802 (verified non-dead) |

### Drawer 주요 실험
⚠️ **cw5000+ 체크포인트는 전부 dead (output layer=0)**. cw1000만 actor 학습됨.

| Dir | scale | l2 | CW | Best SR | Last SR | W_max | Notes |
|-----|-------|-----|----|---------|---------|-------|-------|
| `d2only_as0.05_cw1000_delta_H_*` | 0.05 | 0.01 | 1000 | **100%** | **100%** | 0.802 | ✅ **BEST** |
| `drawer_td3_192235` | 0.1 | 0.0 | 1000 | 93.3% | 93.3% | 0.458 | ✅ **2nd** D2/D3/D4 multi-drawer |
| `d2only_as0.1_cw1000_delta_I_*` | 0.1 | 0.01 | 1000 | 85% | 35% | 0.723 | ✅ **3rd** D2 only, 불안정 |


---

## Stack Cube — **65% SR** (v1) / **60% SR @ 30-pos** (v2, 학습 중)

### v1: 단일 포지션 (ActionScaler, per-task 구버전)

| Key | Value |
|-----|-------|
| **Best checkpoint** | `outputs/stack_rl/sp5_v2_sparse_best/checkpoints/best.pt` |
| **GR00T checkpoint** | `~/DATA/INTERN/training/groot_stack_sim/checkpoint-100000` |
| **Offline data** | `outputs/offline_stack_66ep` (18,807 samples, 66 files) |
| **action_scale** | 0.1 |
| **Code version** | per-task |
| **ActionScaler stats key** | `stack` |
| **Train positions** | 단일 고정 포지션 |

#### v1 Top 3
| # | Dir | AS | Best SR | Last SR | W_max | Notes |
|---|-----|----|---------|---------|-------|-------|
| 🥇 | `stack_rl/sp5_v2_sparse_best` | 0.1 | **65.0%** | 40.0% | 0.921 | 📦 archive |
| 🥈 | `stack_rl/dn10_v2_a05_l01` | 0.05 | 60.0% | 30.0% | 4.183 | 📦 archive |
| 🥉 | `stack_rl/dn7_v2_drawer_best` | 0.05 | 60.0% | 25.0% | 4.847 | ✅ **3rd** |

### v2: 30 포지션 multi-position (per-task, 학습 중 ~36k steps)

| Key | Value |
|-----|-------|
| **GR00T checkpoint** | `~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000` |
| **Offline data** | `outputs/offline_stack_66ep` |
| **Train positions** | `configs/stack_train_30.json` (30개, 순차 할당) |
| **Eval positions** | `configs/stack_eval_20.json` (20개 × 1 episode) |
| **Code version** | per-task (`train_residual_td3_mujoco_stack.py`) |

#### v2 Top 2 (학습 진행 중, ~36k/500k steps)
| # | Dir (SLURM job) | AS | L2 | γ | OF | Best SR | Last SR | Notes |
|---|-----------------|----|----|---|-----|---------|---------|-------|
| 🥇 | `stack_sweep/stk_2` (6443) | 0.1 | 0.0 | 0.95 | 0.5 | **60%** | 55% | ✅ 상승 추세, 34k에서 60% 재달성 |
| 🥈 | `stack_sweep/stk_6` (6447) | 0.05 | 0.0 | 0.95 | 0.5 | **60%** | 40% | 🟡 12k peak 후 40-55% |

> ⚠️ v2는 30개 다양한 포지션 (white-green 거리 0.16~0.50m)에서 평가하므로, v1 단일 포지션 65%보다 v2 60%가 실질적으로 더 어려운 세팅에서의 성능.

---

## Pick and Place (PnP) — **75% SR** (ActionScaler version)

| Key | Value |
|-----|-------|
| **Best checkpoint** | `outputs/pnp_rl_5694/checkpoints/best.pt` |
| **GR00T checkpoint** | `~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000` |
| **Offline data** | `outputs/offline_data/pnp_66ep_dense_v3` (256,998 samples, 640 files) |
| **action_scale** | 0.1 |
| **action_l2_reg** | 0.1 |
| **Code version** | **unified** (`train_residual_td3_unified.py --task pnp`) |
| **ActionScaler stats key** | `pnp` |

### PnP Top 3 (ActionScaler=True only)
| # | Dir | AS | L2 | Best SR | Last SR | W_max | Notes |
|---|-----|----|----|---------|---------|-------|-------|
| 🥇 | `pnp_rl_5694` | 0.1 | 0.1 | **75.0%** | 55.0% | 5.666 | ✅ **BEST** |
| 🥈 | `pnp_rl_5693` | 0.2 | 0.1 | 65.0% | 35.0% | 5.351 | ✅ **2nd** |
| 🥉 | `pnp_rl_5692` | 0.1 | 0.01 | 50.0% | **50.0%** | 0.561 | ✅ **3rd** 가장 안정적 |

---

## Stand Cup — **45% SR** (ActionScaler version)

| Key | Value |
|-----|-------|
| **Best checkpoint** | `outputs/cup_rl_5291/checkpoints/best.pt` |
| **GR00T checkpoint** | `~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000` |
| **Offline data** | `outputs/offline_cup_batch` (267,696 samples, 802 files) |
| **action_scale** | 0.1 |
| **action_l2_reg** | 1.0 |
| **Code version** | per-task (구버전 cup script) |
| **ActionScaler stats key** | `cup` |

### Cup Top 3 (ActionScaler=True only)
| # | Dir | AS | L2 | Best SR | Last SR | W_max | Notes |
|---|-----|----|----|---------|---------|-------|-------|
| 🥇 | `cup_rl_5291` | 0.1 | 1.0 | **45.0%** | 20.0% | 2.605 | ✅ **BEST** |
| 🥈 | `cup_rl_5466` | 0.1 | 1.0 | 45.0% | 20.0% | 6.117 | ✅ **2nd** |
| 🥉 | `cup_rl_5286` | 0.1 | 1.0 | 40.0% | **35.0%** | 4.785 | ✅ **3rd** 가장 안정적 |

---

## ActionScaler Stats Reference (`action_scaler_stats.json`)

| Task | Key | Samples | Files | Offline Dir |
|------|-----|---------|-------|-------------|
| Lift | `lift` | 149,614 | 320 | `~/Repos/Intern/outputs/offline_data/gr00t_sim_66ep/hard` |
| Drawer | `drawer` | 16,810 | 90 | `outputs/offline_drawer_zgate_100k` |
| PnP | `pnp` | 256,998 | 640 | `outputs/offline_data/pnp_66ep_dense_v3` |
| Stack | `stack` | 18,807 | 66 | `outputs/offline_stack_66ep` |
| Cup | `cup` | 267,696 | 802 | `outputs/offline_cup_batch` |

---

## SLURM Scripts

| Task | Script | Partition |
|------|--------|-----------|
| All (unified) | `scripts/slurm_train_td3.sh` | core-extra / core-own |
| Cup (old) | `scripts/slurm_train_cup_td3.sh` | core-own |
| Drawer (D2) | per-experiment scripts | core-own |

## Training Code

| Version | Script | Supports |
|---------|--------|----------|
| **Unified (신버전)** | `resfit/rl_finetuning/scripts/train_residual_td3_unified.py` | ActionScaler, chunk_sync, all tasks |
| Per-task (구버전) | `resfit/rl_finetuning/scripts/train_residual_td3.py` | ActionScaler, per-task only |
| Cup (구버전) | `resfit/rl_finetuning/scripts/train_residual_td3_cup.py` | ActionScaler, cup only |
