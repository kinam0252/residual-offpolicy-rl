#!/bin/bash
#SBATCH --job-name=drawer-eval-compare
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/slurm-%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/slurm-%j.out

set -e

. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/nas_main/.cache/huggingface
export DS_BUILD_OPS=0
export CUDA_HOME=~/fake_cuda

NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}

cd ~/Repos/Intern/residual-offpolicy-rl

python scripts/eval_drawer_compare.py \
    --best_checkpoint outputs/drawer_td3_v5_20260501_192236/checkpoints/best.pt \
    --groot_checkpoint ~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000 \
    --offline_data_dir outputs/offline_drawer_zgate_100k \
    --active_drawers 2 3 4 \
    --action_scale 0.1 \
    --use_action_scaler \
    --num_episodes 20 \
    --device cuda
