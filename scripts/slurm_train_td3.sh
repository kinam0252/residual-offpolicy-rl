#!/bin/bash
#SBATCH --job-name=unified_rl
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/.local/lib:$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
# GPU: SLURM sets CUDA_VISIBLE_DEVICES automatically (remapped to 0).
# Do NOT override with SLURM_JOB_GPUS (physical index) — that breaks the remapping.
echo "[gpu] SLURM_JOB_GPUS=$SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-}"

cd ~/Repos/Intern/residual-offpolicy-rl

# ══════════════════════════════════════════════════════════════════
# Task selection (REQUIRED)
# ══════════════════════════════════════════════════════════════════
TASK=${TASK:-cup}

# ── Task-specific defaults ──
case "$TASK" in
  cup)
    GROOT_CKPT=${GROOT_CKPT:-~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000}
    OFFLINE_DIR=${OFFLINE_DIR:-outputs/offline_cup_batch}
    NUM_ENVS=${NUM_ENVS:-27}
    TOTAL_STEPS=${TOTAL_STEPS:-500000}
    GAMMA=${GAMMA:-0.95}
    ACTION_SCALE=${ACTION_SCALE:-0.1}
    ACTION_L2=${ACTION_L2:-1.0}
    OFFLINE_FRAC=${OFFLINE_FRAC:-0.75}
    REWARD_TYPE=${REWARD_TYPE:-dense}
    TASK_EXTRA=""
    ;;
  pnp)
    GROOT_CKPT=${GROOT_CKPT:-~/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-100000}
    OFFLINE_DIR=${OFFLINE_DIR:-outputs/offline_data/pnp_66ep_dense_v3}
    NUM_ENVS=${NUM_ENVS:-30}
    TOTAL_STEPS=${TOTAL_STEPS:-500000}
    GAMMA=${GAMMA:-0.99}
    ACTION_SCALE=${ACTION_SCALE:-0.2}
    ACTION_L2=${ACTION_L2:-0.01}
    OFFLINE_FRAC=${OFFLINE_FRAC:-0.5}
    REWARD_TYPE=${REWARD_TYPE:-dense_v3}
    # PnP-specific
    EPISODE_POS=${EPISODE_POS:-configs/pnp_sim33ep_positions.json}
    TASK_EXTRA="--episode_positions_file $EPISODE_POS --max_episode_steps 500 --eval_num_episodes 1 --eval_num_envs 20"
    if [ -n "$RANDOM_CUBE_RANGE" ]; then
        TASK_EXTRA="$TASK_EXTRA --random_cube_range '$RANDOM_CUBE_RANGE'"
    fi
    ;;
  lift)
    GROOT_CKPT=${GROOT_CKPT:-~/DATA/INTERN/training/gr00t_sim_66ep/checkpoint-300000}
    OFFLINE_DIR=${OFFLINE_DIR:-outputs/offline_cube_batch}
    NUM_ENVS=${NUM_ENVS:-10}
    TOTAL_STEPS=${TOTAL_STEPS:-50000}
    GAMMA=${GAMMA:-0.99}
    ACTION_SCALE=${ACTION_SCALE:-0.1}
    ACTION_L2=${ACTION_L2:-10.0}
    OFFLINE_FRAC=${OFFLINE_FRAC:-0.5}
    REWARD_TYPE=${REWARD_TYPE:-dense}
    TASK_EXTRA=""
    ;;
  stack)
    GROOT_CKPT=${GROOT_CKPT:-~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000}
    OFFLINE_DIR=${OFFLINE_DIR:-outputs/offline_data/stack_75k}
    NUM_ENVS=${NUM_ENVS:-30}
    TOTAL_STEPS=${TOTAL_STEPS:-500000}
    GAMMA=${GAMMA:-0.95}
    ACTION_SCALE=${ACTION_SCALE:-0.1}
    ACTION_L2=${ACTION_L2:-0.0}
    OFFLINE_FRAC=${OFFLINE_FRAC:-0.5}
    REWARD_TYPE=${REWARD_TYPE:-dense}
    REWARD_RELABEL=${REWARD_RELABEL:-computed}
    REWARD_CONFIG=${REWARD_CONFIG:-configs/reward_stack.yaml}
    EPISODE_POS=${EPISODE_POS:-configs/stack_train_30.json}
    EVAL_POS=${EVAL_POS:-configs/stack_eval_20.json}
    TASK_EXTRA="--episode_positions_file $EPISODE_POS --eval_positions_file $EVAL_POS --eval_num_envs 20 --eval_num_episodes 1 --max_episode_steps 1000 --offline_reward_relabel $REWARD_RELABEL --reward_config $REWARD_CONFIG"
    ;;
  drawer)
    GROOT_CKPT=${GROOT_CKPT:-~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000}
    OFFLINE_DIR=${OFFLINE_DIR:-outputs/offline_drawer_zgate_100k}
    NUM_ENVS=${NUM_ENVS:-10}
    TOTAL_STEPS=${TOTAL_STEPS:-50000}
    GAMMA=${GAMMA:-0.99}
    ACTION_SCALE=${ACTION_SCALE:-0.1}
    ACTION_L2=${ACTION_L2:-10.0}
    OFFLINE_FRAC=${OFFLINE_FRAC:-0.3}
    REWARD_TYPE=${REWARD_TYPE:-dense}
    TASK_EXTRA="--max_episode_steps 500"
    ;;
  *)
    echo "ERROR: Unknown TASK=$TASK (use: cup, pnp, lift, stack, drawer)"
    exit 1
    ;;
esac

WANDB_NAME=${WANDB_NAME:-${TASK}_rl}

# ══════════════════════════════════════════════════════════════════
# Launch unified training
# ══════════════════════════════════════════════════════════════════
python3 resfit/rl_finetuning/scripts/train_residual_td3_unified.py \
    --task "$TASK" \
    --groot_checkpoint "$GROOT_CKPT" \
    --offline_data_dir "$OFFLINE_DIR" \
    --num_envs $NUM_ENVS \
    --total_timesteps $TOTAL_STEPS \
    --action_scale $ACTION_SCALE \
    --action_l2_reg $ACTION_L2 \
    --gamma $GAMMA \
    --offline_fraction $OFFLINE_FRAC \
    --reward_type $REWARD_TYPE \
    --use_action_scaler \
    --no_action_clamp \
    --chunk_sync \
    --async_eval \
    --eval_interval 2000 \
    --eval_num_episodes 1 \
    --eval_num_envs 15 \
    --checkpoint_interval 10000 \
    --wandb_mode online \
    --wandb_name "$WANDB_NAME" \
    --output_dir "outputs/${TASK}_rl_${SLURM_JOB_ID}" \
    $TASK_EXTRA \
    "$@"
