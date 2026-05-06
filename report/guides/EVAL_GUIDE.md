# Standalone Eval Guide

## Scripts

### `scripts/standalone_eval_drawer.py`
Standalone evaluation for Close Drawer checkpoints. Loads GR00T + residual agent and runs eval episodes.

**Key arguments:**
- `checkpoint` (positional): Path to `.pt` checkpoint file
- `--base_only`: Zero out residual actions (eval GR00T base policy only)
- `--num_envs 20`: Number of parallel environments (default: 20)
- `--episodes 1`: Episodes per env (default: 10). With 20 envs × 1 ep = 20 total episodes
- `--stddev 0.0`: Eval noise (default: 0.0)
- `--action_scale 0.05`: Residual action scale (must match training config)
- `--active_drawers 2`: Which drawer(s) to eval (default: [2])
- `--max_episode_steps 500`: Max steps per episode
- `--device cuda`: Device

## Running on Login Node

### Required Environment Variables

```bash
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=7  # or any free GPU
```

### Required LD_LIBRARY_PATH

The login node needs extra library paths for flash_attn (CXXABI_1.3.15) and EGL:

```bash
export LD_LIBRARY_PATH="/home/nas_main/junhahyung/miniconda3/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cufft/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cusolver/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cusparse/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/nccl/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/lib/x86_64-linux-gnu/lib-compat:$HOME/.local/lib/gl:$LD_LIBRARY_PATH"
```

**Why**: System libstdc++ (6.0.30) lacks `CXXABI_1.3.15` needed by `flash_attn_2_cuda`. We borrow a newer libstdc++ (6.0.34) from conda.

### One-liner Examples

```bash
cd ~/Repos/Intern/residual-offpolicy-rl

# --- Close Drawer: Residual TD3 eval ---
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 \
LD_LIBRARY_PATH="/home/nas_main/junhahyung/miniconda3/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cufft/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cusolver/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/cusparse/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/nccl/lib:$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/lib/x86_64-linux-gnu/lib-compat:$HOME/.local/lib/gl:$LD_LIBRARY_PATH" \
$HOME/.venvs/groot/bin/python3 scripts/standalone_eval_drawer.py \
  <CHECKPOINT_PATH> --episodes 1 --num_envs 20

# --- Close Drawer: Base policy only (GR00T, no residual) ---
# Same as above but add --base_only
$HOME/.venvs/groot/bin/python3 scripts/standalone_eval_drawer.py \
  <CHECKPOINT_PATH> --base_only --episodes 1 --num_envs 20
```

### Troubleshooting

| Error | Fix |
|-------|-----|
| `CXXABI_1.3.15 not found` | Add conda libstdc++ to LD_LIBRARY_PATH (see above) |
| `EGL not initialized` | Set `PYOPENGL_PLATFORM=egl` and `MUJOCO_GL=egl` |
| `srun: not allowed` | Don't use srun — run python directly on login node |
| `CUDA out of memory` | Use a different GPU: `CUDA_VISIBLE_DEVICES=<N>` |
