# Active Experiments & Progress

> Last updated: 2026-05-03 02:44 KST

## Running SLURM Jobs (8/8 core-own)

| Job | Name | Config | Node | Started | Notes |
|-----|------|--------|------|---------|-------|
| 4730 | stk_sp5 | sparse, as=0.1, L2=0.1, γ=0.99, n=3 | w7 | ~18:00 | Old 50K limit, best SR=59% @10K |
| 4865 | stk_r05v2 | dense, as=0.1, L2=1.0 | w8 | 01:47 | 500K, wandb online |
| 4866 | stk_dn7v2 | dense, as=0.05, L2=0.01 (drawer best) | w6 | 01:47 | 500K, wandb online |
| 4867 | stk_dn8v2 | dense, as=0.1, L2=0.01 | w7 | 01:47 | 500K, wandb online |
| 4870 | stk_sp2v2 | sparse, as=0.1, γ=0.99, n=5 | w3 | 01:47 | 500K, wandb online |
| 4871 | stk_sp4v2 | sparse, as=0.05, γ=0.99, n=3 | w3 | 01:47 | 500K, wandb online |
| 4872 | stk_dn10v2 | dense, as=0.05, L2=0.1 | w4 | 01:47 | 500K, wandb online |
| 4873 | stk_sp5v2 | sparse, as=0.1, L2=0.1, γ=0.99, n=3 | w4 | 01:47 | 500K remake of sp5 |

## Completed Results

### Close Drawer (D2)
- **Best config**: as=0.05, L2=0.01, delta reward, D2-only, critic_warmup=1000
- Training peak: 100% SR (step 50001)
- Standalone eval: best.pt=80%, final_step50001.pt=**95%**, base_only=50%
- **Residual adds +30~45%p over base GR00T**
- Videos: `videos/drawer/` (residual 5/5=100%, base 3/5=60%)

### Stack Cube (current best: sp5)
- **sp5 best checkpoint**: SR=59% @step 10K (sparse, as=0.1, L2=0.1)
- Standalone eval: **residual 13/20=65%**, base_only 3/9≈33%
- **Residual adds +32%p over base GR00T**
- Videos: `videos/stack/` (per-env with SUCCESS/FAIL naming)

## Wandb Projects
- Drawer: `draftrec/mujoco-drawer-residual-td3`
- Stack: `draftrec/mujoco-franka-stack-residual-td3`
- v2 jobs are **online** (real-time monitoring)

## Key Findings
1. **Drawer solved**: 95% SR with residual (vs 50% base), ~124 steps avg
2. **Stack in progress**: 65% SR at 10K steps, 500K runs just started
3. **Sparse reward** (sp5) currently beating dense configs for stack
4. **Action scale** 0.1 > 0.05 for stack (more exploration needed)
5. Stack cube is harder — base policy only ~33% vs drawer 50%

## Next Steps
- [ ] Monitor v2 500K runs — first meaningful eval at ~2K steps (ETA ~04:00)
- [ ] If stack plateaus at 500K, consider reward redesign (delta reward, remove approach/grasp saturation)
- [ ] Run base_only stack video for full 20 envs (previous was preempted at 9/20)
- [ ] Extend drawer training past 50K? (final=95%, could reach 100% stable)

## Cluster Notes
- `core-own`: 8 GPU max, priority 100222, most stable — **ALL USED**
- `core-extra`: lower priority (40222), may PENDING
- `core-on-sub`: preemptible (25222), good for short eval jobs
- Login node: 8× B200, use CUDA_VISIBLE_DEVICES + EGL for eval (no srun)
- **CRITICAL**: LD_LIBRARY_PATH must include `~/lib-compat` (libstdc++ 6.0.34) for flash_attn
