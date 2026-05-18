#!/bin/bash
#SBATCH --job-name=unified_rl
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=219G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
# Match EGL device to SLURM-assigned GPU (avoids EGL picking wrong device)
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
# NOTE: async_prefetch=False is set in train/eval scripts to avoid EGL thread crashes
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib:$HOME/lib-compat:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
# Fix wandb TLS: compute nodes lack system CA bundle.
# Use certifi's bundled CA certs which exist on NAS.
CERTIFI_CA=$($HOME/.venvs/groot/bin/python3 -c "import certifi; print(certifi.where())" 2>/dev/null)
export SSL_CERT_FILE="${CERTIFI_CA:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[gpu] SLURM_JOB_GPUS=$SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-}"

cd ~/Repos/Intern/residual-offpolicy-rl

# ══════════════════════════════════════════════════════════════════
# Usage: TASK=pnp sbatch scripts/slurm_train_td3.sh [extra args]
#   or:  CONFIG=configs/tasks/pnp.json sbatch scripts/slurm_train_td3.sh
#
# All task settings come from JSON config (configs/tasks/<task>.json).
# Environment variables override JSON: GAMMA, ACTION_SCALE, etc.
# Extra CLI args after the script are passed through to train.
# ══════════════════════════════════════════════════════════════════

TASK=${TASK:-cup}
CONFIG=${CONFIG:-configs/tasks/${TASK}.json}
WANDB_NAME=${WANDB_NAME:-${TASK}_rl}

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Build CLI override list from env vars (only if explicitly set)
OVERRIDES=""
[ -n "$GAMMA" ] && OVERRIDES="$OVERRIDES --gamma $GAMMA"
[ -n "$ACTION_SCALE" ] && OVERRIDES="$OVERRIDES --action_scale $ACTION_SCALE"
[ -n "$ACTION_L2" ] && OVERRIDES="$OVERRIDES --action_l2_reg $ACTION_L2"
[ -n "$OFFLINE_FRAC" ] && OVERRIDES="$OVERRIDES --offline_fraction $OFFLINE_FRAC"
[ -n "$REWARD_TYPE" ] && OVERRIDES="$OVERRIDES --reward_type $REWARD_TYPE"
[ -n "$NUM_ENVS" ] && OVERRIDES="$OVERRIDES --num_envs $NUM_ENVS"
[ -n "$TOTAL_STEPS" ] && OVERRIDES="$OVERRIDES --total_timesteps $TOTAL_STEPS"
[ -n "$GROOT_CKPT" ] && OVERRIDES="$OVERRIDES --groot_checkpoint $GROOT_CKPT"
[ -n "$OFFLINE_DIR" ] && OVERRIDES="$OVERRIDES --offline_data_dir $OFFLINE_DIR"
[ -n "$RESIDUAL_POS_SCALE" ] && OVERRIDES="$OVERRIDES --residual_pos_scale $RESIDUAL_POS_SCALE"
[ -n "$RESIDUAL_GRIP_SCALE" ] && OVERRIDES="$OVERRIDES --residual_grip_scale $RESIDUAL_GRIP_SCALE"
[ -n "$RESIDUAL_ROT_SCALE" ] && OVERRIDES="$OVERRIDES --residual_rot_scale $RESIDUAL_ROT_SCALE"
# Object state augmentation
[ -n "$OBS_NOISE_MAX" ] && OVERRIDES="$OVERRIDES --use_obs_noise --obs_noise_max $OBS_NOISE_MAX"
[ -n "$OBS_DROPOUT_PROB" ] && OVERRIDES="$OVERRIDES --use_obs_dropout --obs_dropout_prob $OBS_DROPOUT_PROB"
[ -n "$DISABLE_OBJECT_STATE" ] && OVERRIDES="$OVERRIDES --disable_object_state"
# Drawer-specific
[ -n "$ACTIVE_DRAWERS" ] && OVERRIDES="$OVERRIDES --active_drawers $ACTIVE_DRAWERS"
[ -n "$CONTACT_Z_GATE" ] && OVERRIDES="$OVERRIDES --contact_z_gate"
# Network size
[ -n "$ACTOR_HIDDEN" ] && OVERRIDES="$OVERRIDES --actor_hidden_dim $ACTOR_HIDDEN"
[ -n "$CRITIC_HIDDEN" ] && OVERRIDES="$OVERRIDES --critic_hidden_dim $CRITIC_HIDDEN"

# ══════════════════════════════════════════════════════════════════
# Launch unified training
# ══════════════════════════════════════════════════════════════════
python3 resfit/rl_finetuning/scripts/train_residual_td3_unified.py \
    --config "$CONFIG" \
    --use_action_scaler \
    --no_action_clamp \
    --chunk_sync \
    --async_eval \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --wandb_mode online \
    --wandb_name "$WANDB_NAME" \
    --output_dir "outputs/${TASK}_rl_${SLURM_JOB_ID}" \
    $OVERRIDES \
    "$@"
