# Active Experiments & Progress

> Last updated: 2026-05-04 06:00 KST

## Best Results Summary (All Tasks)

| Task | Best SR | Base SR | **Improvement** | Config | Code | Notes |
|------|---------|---------|----------------|--------|------|-------|
| **Cube Lift** | **100%** | 18% | **+82%p** | ActionScaler, s=0.1, l2=5.0 | per-task | Historical best |
| **Close Drawer** | **100%** | 50% | **+50%p** | ActionScaler, s=0.05, l2=0.01, delta reward | per-task (D2-only, job 4394) | 25K/75K ckpt, eval 95% |
| **Stack Cube** | **80%** | 57% | **+23%p** | ActionScaler, s=0.1, l2=1.0, dense | per-task (hard set) | Historical sweep |
| **PnP** | **50%** | ~40% | **+10%p** | ActionScaler, s=0.1, l2=0.01, dense_v3 | **unified** (job 5692) | 4K/500K, 초기 |
| **Stand Cup** | **40%** | ~30% | **+10%p** | dense, s=0.1, l2=1.0 | per-task (job 5466) | 34K/500K, 정체 |

**Pattern**: Improvement ∝ (1 - Base SR). Lower base → bigger residual gain.

## Running SLURM Jobs

### PnP (Unified Code, core-extra)
| Job | Name | Config | Node | SR Trend | Speed |
|-----|------|--------|------|----------|-------|
| 5691 | pnp_unif | s=0.2, l2=0.01 (baseline) | w2 | 65→10→40→35% | 2.2 sps |
| 5692 | pnp_s01 | **s=0.1**, l2=0.01 | - | 50→45→**50%** ✅ | 2.1 sps |
| 5693 | pnp_l201 | s=0.2, **l2=0.1** | w6 | 35→25→**50%** ✅ | 2.3 sps |
| 5694 | pnp_s01l2 | **s=0.1, l2=0.1** | w3 | 30→**45%** | 2.1 sps |

### Cup (Per-task Code, core-own / core-on-sub)
| Job | Config | Step | SR (latest) | Speed | Notes |
|-----|--------|------|-------------|-------|-------|
| 5466 | dense, s=0.1, l2=1.0 | 34K | 20% | 1.7 sps | core-own, 정체 |
| 5286 | sparse, s=0.1, l2=1.0 | 56K | 30% | 1.0 sps | core-own, 정체 |
| 5295 | sparse, s=0.05, l2=1.0 | 42K | 25% | 0.7 sps | sub, 정체 |
| 5296 | sparse, s=0.1, l2=10.0 | 41K | 25% | 0.7 sps | sub, 정체 |

### Stack Cube (Per-task Code, core-own)
| Job | Config | Step | SR (latest) | Peak SR | Notes |
|-----|--------|------|-------------|---------|-------|
| 4865 | dense, s=0.1, l2=1.0 | 74K | 35% | **55%** @28K | 고점 후 하락 |
| 4871 | sparse, s=0.05, l2=1.0 | 84K | 25% | 40% @32K | 불안정 |

### PnP Old (Per-task Code, core-own) — 구버전, no ActionScaler
| Job | Step | SR (latest) | Notes |
|-----|------|-------------|-------|
| 4866 | 4K | 31% | 초기 |
| 4872 | 18K | **4%** 🔴 | SR 붕괴 (27%→4%) |
| 4873 | 13K | 19% | 하락 중 |

## Completed Results

### Close Drawer (D2)
- **Best config**: as=0.05, L2=0.01, delta reward, D2-only, critic_warmup=1000
- Training peak: **100% SR** (step 25K, 75K checkpoints)
- Standalone eval: best.pt=80%, final_step50001.pt=**95%**, base_only=50%
- **Residual adds +30~50%p over base GR00T**
- Videos: `videos/drawer/` (residual 5/5=100%, base 3/5=60%)

### Cube Lift (Historical)
- **Best config**: ActionScaler, s=0.1, l2=5.0
- **100% SR** (from 18% base) — largest improvement across all tasks

### Stack Cube (Historical - Hard set)
- **Best config**: ActionScaler, s=0.1, l2=1.0
- **80% SR** (from 57% base)

## Key Findings
1. **Residual RL improvement ∝ base SR gap**: Lift(18%→100%) > Drawer(50%→100%) > Stack Hard(57%→80%)
2. **ActionScaler is critical** — PnP unified (with ActionScaler) 50% vs old PnP (without) 4-19%
3. **Unified code 2-3× faster** — chunk_sync + correct GPU isolation: 2.1-2.3 sps vs 0.7-1.0 sps
4. **Cup all configs plateau at 20-30%** — 40K+ steps, no improvement over base
5. **PnP s=0.1 most stable** — consistently 50% SR in early training
6. **Drawer solved**: delta reward + conservative residual (s=0.05) key to success
7. **GPU isolation fix**: SLURM auto-sets CUDA_VISIBLE_DEVICES — do NOT override with SLURM_JOB_GPUS

## Unified Codebase
- `resfit/rl_finetuning/scripts/train_residual_td3_unified.py` — all tasks via `--task`
- `resfit/rl_finetuning/scripts/eval_async_unified.py` — unified async eval
- `scripts/slurm_train_td3.sh` — unified SLURM with task-specific defaults
- `resfit/rl_finetuning/configs/task_configs.py` — per-task config dataclass
- `resfit/rl_finetuning/wrappers/mujoco_residual_wrapper_unified.py` — unified wrapper

## Wandb Projects
- Drawer: `draftrec/mujoco-drawer-residual-td3`
- Stack: `draftrec/mujoco-franka-stack-residual-td3`
- Cup: `draftrec/mujoco-franka-cup-residual-td3` (구버전 jobs)
- PnP: `draftrec/mujoco-franka-pnp-residual-td3` (unified jobs, online)

## Cluster Notes
- `core-own`: 8 GPU max, priority 100222, most stable
- `core-extra`: lower priority (40222), may PENDING but no preemption
- `core-on-sub`: preemptible (25222), good for short jobs only
- **GPU isolation**: Do NOT `export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}` — SLURM handles it automatically
- **CRITICAL**: LD_LIBRARY_PATH must include `~/lib-compat` (libstdc++ 6.0.34) for flash_attn
