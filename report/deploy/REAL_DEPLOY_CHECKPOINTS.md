# Real Robot Deploy — Checkpoint Reference

> Checkpoints actually used for real robot evaluation.
> These may differ from sim-best checkpoints in `BEST_CHECKPOINTS.md`.

---

## Summary

| Task | RL Variant | Action Scale | Clamp | Base SR | Residual SR | Trials (B/R) |
|------|-----------|-------------|-------|---------|-------------|--------------|
| Cube Lift | `v58_smallnet_old` | 0.2 | yes | 71.0% (22/31) | **86.1%** (31/36) | 31 / 36 |
| Pick & Place | `pnp_rl_12030_best` | 0.1 | no | 76.2% (16/21) | 57.1% (20/35) | 21 / 35 |
| Close Drawer | `d2only_as005_cw1000_delta_H` | 0.05 | — | 62.5% (30/48) | **96.9%** (31/32) | 48 / 32 |
| Stack Cube | `dn7_v2_drawer_best` | 0.05 | no | 47.6% (20/42) | 38.5% (20/52) | 42 / 52 |
| Stand Cup | `cup_rl_5286` | 0.1 | no | 0.0% (0/12) | 9.5% (2/21) | 12 / 21 |

---

## Per-Task Details

### 1. Cube Lift

| Key | Value |
|-----|-------|
| **GR00T base** | `checkpoints/cube_lift/groot_32ep/100k` |
| **RL checkpoint** | `checkpoints/cube_lift/rl_best/_archive/v58_smallnet_old/checkpoints/best.pt` |
| **Variant** | `v58_smallnet_old` |
| **Action scale** | 0.2 |
| **no_action_clamp** | false (action clamped) |
| **Epoch tag** | 32ep |

> ⚠️ Sim best는 `as_sweep_hard_as01_noL2` (as=0.1)이지만, real deploy에는 구버전 `v58_smallnet_old` (as=0.2) 사용.

---

### 2. Pick & Place

| Key | Value |
|-----|-------|
| **GR00T base** | `checkpoints/pick_and_place/groot_pnp_66ep/100k` |
| **RL checkpoint** | `checkpoints/pick_and_place/rl_unified_pnp/pnp_rl_12030_best.pt` |
| **Variant** | `pick_and_place` (unified training) |
| **Action scale** | 0.1 |
| **no_action_clamp** | true |
| **Epoch tag** | 66ep |

> ⚠️ Sim best는 `pnp_rl_5694` (as=0.1, L2=0.1)이지만, real deploy에는 `pnp_rl_12030_best` 사용.
> 소수 2 trials은 action_scale=0.05로 테스트됨.

---

### 3. Close Drawer

| Key | Value |
|-----|-------|
| **GR00T base** | `checkpoints/close_drawer/groot_drawer_real_33ep/100k` |
| **RL checkpoint** | `checkpoints/close_drawer/rl_best/d2only_as005_cw1000_delta_H/default/checkpoints/best.pt` |
| **Variant** | `d2only_as005_cw1000_delta_H` |
| **Action scale** | 0.05 |
| **Reward type** | delta |
| **Epoch tag** | 33ep |
| **Train source** | `outputs/d2only_td3_as0.05_L20.01_cw1000_delta_H_20260502_095333` |

> ⚠️ trial_meta.json이 없는 옛 포맷. Checkpoint는 rl_best/ 내 2개 중 `d2only` 선택 확인.
> 별도로 `groot_closedrawer_base_augmented_60ep` (aug_base 8 trials)도 있으나 주요 평가 아님.

---

### 4. Stack Cube

| Key | Value |
|-----|-------|
| **GR00T base** | `checkpoints/stack_cube/groot_stack_real_66ep/100k` |
| **RL checkpoint** | `checkpoints/stack_cube/rl_best/dn7_v2_drawer_best/checkpoints/best.pt` |
| **Variant** | `dn7_v2_drawer_best` |
| **Action scale** | 0.05 |
| **no_action_clamp** | true |
| **Epoch tag** | 66ep |

> Sim best는 `sp5_v2_sparse_best` (as=0.1, 65% SR)이지만, real deploy에는 `dn7_v2_drawer_best` (as=0.05) 사용.
> 소수 trials: `sp5_v2_sparse_best` (10), `dn10_v2_a05_l01` (5), `stk_2_v2_30pos` (1).

---

### 5. Stand Cup

| Key | Value |
|-----|-------|
| **GR00T base** | `checkpoints/stand_cup/groot_cup_real_27ep/100k` |
| **RL checkpoint** | `checkpoints/stand_cup/rl_best/cup_rl_5286/checkpoints/best.pt` |
| **Variant** | `cup_rl_5286` |
| **Action scale** | 0.1 |
| **no_action_clamp** | true |
| **Epoch tag** | 27ep |

> Sim best는 `cup_rl_6302` (50% SR)이지만, real deploy에는 `cup_rl_5286` 사용.
> 소수 trials: `cup_rl_5291` (4), `cup_rl_6302` (1), `cup_rl_5466` (1).

---

## Notes

- **Action scale**: RL output에 곱해지는 스케일. 0.05~0.2 범위. 작을수록 보수적.
- **no_action_clamp**: true면 residual output clamp 없음 (unified 버전). false면 [-0.1, 0.1] clamp (구버전).
- **Epoch tag**: GR00T base policy의 학습 에폭 (32ep, 66ep 등).
- Sim에서 best와 real deploy checkpoint이 다른 이유: sim-to-real transfer 시 다른 checkpoint이 더 잘 동작하는 경우가 있음.
