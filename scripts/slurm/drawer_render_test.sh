#!/bin/bash
#SBATCH --job-name=drawer_render_test
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=core-extra

set -euo pipefail
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib:$HOME/lib-compat:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

python3 resfit/rl_finetuning/scripts/save_raw_images_drawer.py
