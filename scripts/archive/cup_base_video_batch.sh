#!/bin/bash
#SBATCH --job-name=cup_video
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cup_video_%A_%a.out
#SBATCH --error=logs/cup_video_%A_%a.err
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

echo "=== Cup Base Video: chunk ${SLURM_ARRAY_TASK_ID}/7 ==="

python3 scripts/cup_base_video_batch.py \
    --chunk_id ${SLURM_ARRAY_TASK_ID} \
    --num_chunks 8 \
    --out_dir outputs/cup_base_videos

echo "=== Done chunk ${SLURM_ARRAY_TASK_ID} ==="
