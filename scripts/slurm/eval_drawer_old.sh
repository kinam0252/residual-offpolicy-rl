#!/bin/bash
#SBATCH --job-name=drawer_old_eval
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nccl/lib:$NV/nvjitlink/lib:$HOME/lib-compat:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}"

cd ~/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH="$PWD:$PYTHONPATH"

CKPT="outputs/d2only_td3_as0.05_L20.01_cw1000_delta_H_20260502_095333/checkpoints/best.pt"

# PROJECT_ROOT fix: standalone_eval_drawer.py sets PROJECT_ROOT to scripts/archive/eval/../ = scripts/
# We need project root on path, already done via PYTHONPATH

if [ "${MODE}" = "base" ]; then
    echo "[eval] Running OLD standalone eval - BASE ONLY"
    python3 scripts/archive/eval/standalone_eval_drawer.py "$CKPT" \
        --base_only --episodes 1 --num_envs 20 --active_drawers 2
else
    echo "[eval] Running OLD standalone eval - RESIDUAL"
    python3 scripts/archive/eval/standalone_eval_drawer.py "$CKPT" \
        --episodes 1 --num_envs 20 --active_drawers 2
fi
