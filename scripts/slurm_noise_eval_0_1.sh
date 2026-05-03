#!/bin/bash
#SBATCH --job-name=noise_0_1
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/noise_eval_0_1_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/noise_eval_0_1_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0

cd ~/Repos/Intern/residual-offpolicy-rl

python3 scripts/noise_eval_stack.py \
    outputs/stack_rl/sp5_g99_n3_l2lo/checkpoints/best.pt \
    --noise_std 0.1 \
    --num_envs 20
