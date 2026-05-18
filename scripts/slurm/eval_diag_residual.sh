#!/bin/bash
#SBATCH --job-name=cup_diag_res
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.out

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib:$HOME/lib-compat:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PYTHONPATH=$HOME/Repos/Intern/residual-offpolicy-rl:$PYTHONPATH

cd $HOME/Repos/Intern/residual-offpolicy-rl

# Residual eval: uses RL checkpoint's learned residual
CKPT=${CKPT:-outputs/cup_rl_1961/checkpoints/best.pt}
OUT=${OUT:-outputs/cup_diag_residual}

python3 scripts/eval_diagnostic.py \
    --checkpoint "$CKPT" \
    --num_envs 27 \
    --num_episodes 1 \
    --output_dir "$OUT"
