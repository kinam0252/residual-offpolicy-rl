#!/bin/bash
#SBATCH --job-name=drawer_phys_sweep
#SBATCH --qos=share
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/.local/lib:$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[gpu] SLURM_JOB_GPUS=$SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd ~/Repos/Intern/residual-offpolicy-rl

# ══════════════════════════════════════════════════════════════════
# Drawer physics RL sweep v2 — 75k ckpt, physics_drawer=True, D2+D3
#
# 6 configs exploring key axes:
#   A: drawer_best baseline (scale=0.05, L2=0.01, delta)
#   B: higher scale (0.1) + dense reward
#   C: high L2 regularization (from stack best)
#   D: lower gamma (0.95, from cup/lift best)
#   E: conservative (tiny scale, high L2)
#   F: aggressive (large scale, low L2, high offline)
#
# Train: D2×10, D3×10 = 20 envs
# Eval:  D2×5,  D3×5  = 10 envs
#
# Usage: CONFIG_NAME=A sbatch scripts/slurm/sweep_drawer_physics_v2.sh
# ══════════════════════════════════════════════════════════════════

CONFIG_NAME=${CONFIG_NAME:-A}
CKPT=~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-75000
OFFLINE_DIR=outputs/offline_drawer_physics_75k

case "$CONFIG_NAME" in
    A)
        # Baseline: past drawer best adapted for physics
        SCALE=0.05; L2=0.01; GAMMA=0.99; OFFLINE=0.3; REWARD=delta
        LR_A=3e-4; LR_C=3e-4; WARMUP=1000; STEPS=50000
        TAG="baseline_delta"
        ;;
    B)
        # Dense reward + moderate scale (PnP-inspired)
        SCALE=0.1; L2=0.1; GAMMA=0.99; OFFLINE=0.3; REWARD=dense
        LR_A=3e-4; LR_C=3e-4; WARMUP=2000; STEPS=50000
        TAG="dense_mod"
        ;;
    C)
        # High L2 + low gamma (stack best style)
        SCALE=0.1; L2=5.0; GAMMA=0.95; OFFLINE=0.3; REWARD=dense
        LR_A=3e-4; LR_C=3e-4; WARMUP=2000; STEPS=50000
        TAG="highL2_lowgamma"
        ;;
    D)
        # Cup/Lift style: low gamma, high offline fraction
        SCALE=0.1; L2=1.0; GAMMA=0.95; OFFLINE=0.5; REWARD=dense
        LR_A=3e-4; LR_C=3e-4; WARMUP=2000; STEPS=50000
        TAG="cup_style"
        ;;
    E)
        # Conservative: tiny residual, high regularization
        SCALE=0.02; L2=10.0; GAMMA=0.99; OFFLINE=0.5; REWARD=delta
        LR_A=1e-4; LR_C=3e-4; WARMUP=2000; STEPS=50000
        TAG="conservative"
        ;;
    F)
        # Aggressive: larger residual, low regularization
        SCALE=0.2; L2=0.1; GAMMA=0.95; OFFLINE=0.3; REWARD=dense
        LR_A=3e-4; LR_C=3e-4; WARMUP=2000; STEPS=80000
        TAG="aggressive"
        ;;
    *)
        echo "ERROR: Unknown CONFIG_NAME=$CONFIG_NAME (choose A-F)"
        exit 1
        ;;
esac

echo "[sweep] Config=$CONFIG_NAME ($TAG)"
echo "  scale=$SCALE L2=$L2 gamma=$GAMMA offline=$OFFLINE reward=$REWARD"
echo "  lr_a=$LR_A lr_c=$LR_C warmup=$WARMUP steps=$STEPS"

python3 resfit/rl_finetuning/scripts/train_residual_td3_unified.py \
    --task drawer \
    --groot_checkpoint "$CKPT" \
    --physics_drawer \
    --use_action_scaler \
    --no_action_clamp \
    --chunk_sync \
    --async_eval \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --total_timesteps "$STEPS" \
    --num_envs 20 \
    --active_drawers 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 \
    --eval_num_envs 10 \
    --eval_active_drawers 2 2 2 2 2 3 3 3 3 3 \
    --action_scale "$SCALE" \
    --action_l2_reg "$L2" \
    --gamma "$GAMMA" \
    --actor_lr "$LR_A" \
    --critic_lr "$LR_C" \
    --critic_warmup_steps "$WARMUP" \
    --offline_data_dir "$OFFLINE_DIR" \
    --offline_fraction "$OFFLINE" \
    --reward_type "$REWARD" \
    --wandb_mode online \
    --wandb_name "drawer_phys_${TAG}" \
    --output_dir "outputs/drawer_phys_${TAG}_${SLURM_JOB_ID}" \
    "$@"
