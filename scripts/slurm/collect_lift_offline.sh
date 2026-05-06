#!/bin/bash
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=col_lift
#SBATCH --output=/home/nas_main/kinamkim/slurms/col_lift_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/col_lift_%j.err
#SBATCH --time=02:00:00

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/.local/lib:$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

cd ~/Repos/Intern/residual-offpolicy-rl

# ══════════════════════════════════════════════════════════════════
# Lift offline data collection with state features
# Work-stealing: submit multiple jobs → they share output dir
#
# Usage:
#   # Submit 4 parallel jobs:
#   for i in $(seq 1 4); do sbatch scripts/slurm/collect_lift_offline.sh; done
#
#   # Or with custom args:
#   NUM_POS=30 EPISODES=10 sbatch scripts/slurm/collect_lift_offline.sh
# ══════════════════════════════════════════════════════════════════

POS_FILE=${POS_FILE:-configs/lift_66ep_positions.json}
GROOT_CKPT=${GROOT_CKPT:-~/DATA/INTERN/training/gr00t_sim_66ep/checkpoint-300000}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/offline_data/lift_30pos_states}
NUM_POS=${NUM_POS:-30}
EPISODES=${EPISODES:-10}
BATCH_SIZE=${BATCH_SIZE:-15}
REWARD_TYPE=${REWARD_TYPE:-dense_clipped}

echo "[Lift Offline] Job $SLURM_JOB_ID"
echo "[Lift Offline] Positions: $POS_FILE (first $NUM_POS)"
echo "[Lift Offline] GR00T: $GROOT_CKPT"
echo "[Lift Offline] Output: $OUTPUT_DIR"
echo "[Lift Offline] Episodes/pos: $EPISODES, Batch: $BATCH_SIZE"

python3 scripts/workloads/collect_lift_offline.py \
    --positions_file "$POS_FILE" \
    --groot_checkpoint "$GROOT_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --num_positions "$NUM_POS" \
    --episodes_per_pos "$EPISODES" \
    --batch_size "$BATCH_SIZE" \
    --reward_type "$REWARD_TYPE" \
    --max_episode_steps 300

echo "DONE: Lift offline collection job $SLURM_JOB_ID"
