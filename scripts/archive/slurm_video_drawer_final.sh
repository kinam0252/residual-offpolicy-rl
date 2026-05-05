#!/bin/bash
#SBATCH --job-name=vid_drawer_final
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/vid_drawer_final_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/vid_drawer_final_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0

cd ~/Repos/Intern/residual-offpolicy-rl

python3 scripts/video_eval_drawer.py \
    outputs/d2only_td3_as0.05_L20.01_cw1000_delta_H_20260502_095333/checkpoints/final_step50001.pt \
    --episodes 5 \
    --save_dir videos/drawer \
    --tag residual_final \
    --action_scale 0.05 \
    --active_drawers 2 \
    --max_episode_steps 500
