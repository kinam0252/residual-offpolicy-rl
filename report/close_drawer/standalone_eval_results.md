# Close Drawer — Standalone Eval Results

## 2026-05-02: D2-Only Best Checkpoint vs Base Policy

**Checkpoint**: `outputs/d2only_td3_as0.05_L20.01_cw1000_delta_H_20260502_095333/checkpoints/best.pt`
- Training job: SLURM 4394
- Config: action_scale=0.05, L2=0.01, delta reward, D2-only, critic_warmup=1000
- Best step: 36000 (wandb peak SR=85%)

**Eval setup**: Login node, CUDA_VISIBLE_DEVICES=7, 20 envs × 1 episode = 20 episodes, stddev=0.0

### Results

| Mode | SR | Episodes | Return |
|------|-----|----------|--------|
| **Residual TD3** (best.pt) | **80%** (16/20) | 20 | 1.708 |
| **Base only** (GR00T, residual=0) | **50%** (10/20) | 20 | 1.387 |

### Key Takeaways

1. **Residual RL adds +30%p** over the frozen GR00T base policy (50% → 80%)
2. **Eval SR (80%) slightly lower than training SR (85%)** — expected due to eval stochasticity with small episode count
3. **Base policy SR=50%** — GR00T alone can close drawer D2 half the time, residual consistently improves this
4. **Return improvement**: 1.387 → 1.708 (+23% relative), indicating residual not only succeeds more often but also completes faster (higher delta reward accumulation)

### Reproduction

```bash
# Residual TD3 eval
cd ~/Repos/Intern/residual-offpolicy-rl
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 \
LD_LIBRARY_PATH="/home/nas_main/junhahyung/miniconda3/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cufft/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cusolver/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cusparse/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/nccl/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/lib/x86_64-linux-gnu/lib-compat:$HOME/.local/lib/gl:$LD_LIBRARY_PATH" \
$HOME/.venvs/groot/bin/python3 scripts/standalone_eval_drawer.py \
  outputs/d2only_td3_as0.05_L20.01_cw1000_delta_H_20260502_095333/checkpoints/best.pt \
  --episodes 1 --num_envs 20

# Base policy only (zero residual)
# Same command but add --base_only flag
$HOME/.venvs/groot/bin/python3 scripts/standalone_eval_drawer.py \
  outputs/d2only_td3_as0.05_L20.01_cw1000_delta_H_20260502_095333/checkpoints/best.pt \
  --base_only --episodes 1 --num_envs 20
```

**Note**: `--base_only` monkey-patches `agent.act()` to return zero tensors, so only GR00T base policy drives the robot. The checkpoint path is still required (for env/agent config) but weights are not loaded.
