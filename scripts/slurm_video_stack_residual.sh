#!/bin/bash
#SBATCH --job-name=vid_stack_res
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/vid_stack_res_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/vid_stack_res_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0

cd ~/Repos/Intern/residual-offpolicy-rl

python3 scripts/video_eval_stack.py \
    outputs/stack_rl/sp5_g99_n3_l2lo/checkpoints/best.pt \
    --num_envs 20 \
    --save_dir videos/stack \
    --tag residual_sp5_best \
    --positions_file configs/stack_cube_positions.json
