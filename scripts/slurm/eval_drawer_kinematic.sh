#!/bin/bash
#SBATCH --job-name=drawer_kinematic_eval
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$HOME/.local/lib:$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

cd ~/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH="$PWD:$PYTHONPATH"

# MODE: "base" or "residual"
MODE=${MODE:-base}
DRAWER=${DRAWER:-2}

CKPT_DIR=outputs/d2only_td3_as0.05_L20.01_cw1000_delta_H_20260502_095333/checkpoints
GROOT_CKPT=~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000

echo "[eval] Mode=$MODE, Drawer=D$DRAWER"

if [ "$MODE" = "residual" ]; then
    python3 scripts/eval_drawer_kinematic.py \
        --mode residual \
        --ckpt "$CKPT_DIR/best.pt" \
        --groot_checkpoint "$GROOT_CKPT" \
        --drawer "$DRAWER" \
        --num_envs 20 --episodes 1 ${SERIAL:+--serial}
else
    python3 scripts/eval_drawer_kinematic.py \
        --mode base \
        --groot_checkpoint "$GROOT_CKPT" \
        --drawer "$DRAWER" \
        --num_envs 20 --episodes 1 ${SERIAL:+--serial}
fi
