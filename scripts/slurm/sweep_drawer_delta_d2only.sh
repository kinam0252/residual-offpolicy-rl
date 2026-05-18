#!/bin/bash
#SBATCH --job-name=drawer_d2_sweep
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
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
CERTIFI_CA=$($HOME/.venvs/groot/bin/python3 -c "import certifi; print(certifi.where())" 2>/dev/null)
export SSL_CERT_FILE="${CERTIFI_CA:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

cd ~/Repos/Intern/residual-offpolicy-rl

# ══════════════════════════════════════════════════════════════════
# Drawer delta reward sweep — D2 only (30 train envs)
# ckpt-75k, physics_drawer=True, 500k steps
#
# Usage: CONFIG_NAME=S1 sbatch scripts/slurm/sweep_drawer_delta_d2only.sh
# ══════════════════════════════════════════════════════════════════

CONFIG_NAME=${CONFIG_NAME:-S1}
CKPT=~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-75000
OFFLINE_DIR=outputs/offline_drawer_physics_75k

L2=0.01; GAMMA=0.99; OFFLINE=0.3; REWARD=delta
LR_A=3e-4; LR_C=3e-4; WARMUP=1000; STEPS=500000

case "$CONFIG_NAME" in
    S1)  SCALE=0.05; TAG="d2_s005" ;;
    S2)  SCALE=0.10; TAG="d2_s010" ;;
    S3)  SCALE=0.15; TAG="d2_s015" ;;
    S4)  SCALE=0.20; TAG="d2_s020" ;;
    S5)  SCALE=0.30; TAG="d2_s030" ;;
    S6)  SCALE=0.50; TAG="d2_s050" ;;
    *)
        echo "ERROR: Unknown CONFIG_NAME=$CONFIG_NAME (choose S1-S6)"
        exit 1
        ;;
esac

echo "[sweep] Config=$CONFIG_NAME ($TAG) scale=$SCALE L2=$L2 gamma=$GAMMA steps=$STEPS"

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
    --num_envs 30 \
    --active_drawers 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 \
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
    --wandb_name "drawer_${TAG}" \
    --output_dir "outputs/drawer_${TAG}_${SLURM_JOB_ID}" \
    "$@"
