#!/bin/bash
# Collect offline Stack Cube data using GR00T base policy (new physics)
#SBATCH --job-name=stack_col
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --nodelist=worker-6
#SBATCH --time=04:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/stack_col_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/stack_col_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DS_BUILD_OPS=0
export CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd ~/Repos/Intern/residual-offpolicy-rl

NUM_EPISODES="${NUM_EPISODES:-66}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/offline_stack_66ep}"

echo "[Stack Offline] Episodes: $NUM_EPISODES"
echo "[Stack Offline] Output: $OUTPUT_DIR"
echo "[Stack Offline] Start: $(date)"

python scripts/collect_stack_offline.py \
    --groot_checkpoint ~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000 \
    --num_episodes $NUM_EPISODES \
    --max_episode_steps 300 \
    --output_dir $OUTPUT_DIR \
    --num_envs 1

echo "[Stack Offline] Done: $(date)"
