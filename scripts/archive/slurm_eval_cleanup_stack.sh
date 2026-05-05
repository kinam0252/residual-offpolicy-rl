#!/bin/bash
#SBATCH --job-name=stk_eval
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/stk_eval_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/stk_eval_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/.local/lib:$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
echo "[gpu] SLURM_JOB_GPUS=$SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd ~/Repos/Intern/residual-offpolicy-rl

# Pass experiment names as positional args to this script
# Usage: sbatch scripts/slurm_eval_cleanup_stack.sh stk_on2 stk_on3 stk_on4
python scripts/batch_eval_stack.py --exp_dirs ${EXP_NAMES:-$@}
