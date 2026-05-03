#!/bin/bash
#SBATCH --job-name=cup_collect
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cup_collect_%A_%a.out
#SBATCH --error=logs/cup_collect_%A_%a.err
#SBATCH --array=0-7

set -e
. ~/.venvs/groot/bin/activate
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0

EPISODES_PER_ENV=13
OUTPUT_DIR="outputs/offline_cup_batch/chunk_${SLURM_ARRAY_TASK_ID}"

echo "=== Cup Collect: chunk ${SLURM_ARRAY_TASK_ID}/7 ==="
echo "  Episodes/env: ${EPISODES_PER_ENV}"
echo "  Output: ${OUTPUT_DIR}"

python3 -u resfit/rl_finetuning/scripts/collect_offline_data_cup.py \
    --groot_checkpoint ~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-200000 \
    --num_episodes_per_env ${EPISODES_PER_ENV} \
    --output_dir "${OUTPUT_DIR}" \
    --max_episode_steps 300 \
    --device cuda:0

echo "=== Done chunk ${SLURM_ARRAY_TASK_ID} ==="
