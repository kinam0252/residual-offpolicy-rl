#!/bin/bash
#SBATCH --job-name=lift_base_eval
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.out

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib:$HOME/lib-compat:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES EGL_DEVICE=$MUJOCO_EGL_DEVICE_ID"

cd ~/Repos/Intern/residual-offpolicy-rl

NUM_ENVS=${NUM_ENVS:-20}
NUM_EPISODES=${NUM_EPISODES:-1}
MAX_STEPS=${MAX_STEPS:-300}
GROOT_CKPT=${GROOT_CKPT:-~/DATA/INTERN/training/groot_lift_sim_32ep_100k/checkpoint-100000}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/lift_base_eval_${SLURM_JOB_ID}}

python3 scripts/eval_base_lift.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --max_episode_steps "$MAX_STEPS" \
    --cube_positions_json configs/lift_66ep_positions.json \
    --parallel_envs \
    --output_dir "$OUTPUT_DIR" \
    "$@"
