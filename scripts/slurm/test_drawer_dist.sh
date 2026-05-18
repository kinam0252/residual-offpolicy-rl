#!/bin/bash
#SBATCH --job-name=drawer_dist_test
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=219G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib:$HOME/lib-compat:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
CERTIFI_CA=$($HOME/.venvs/groot/bin/python3 -c "import certifi; print(certifi.where())" 2>/dev/null)
export SSL_CERT_FILE="${CERTIFI_CA:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "[gpu] SLURM_JOB_GPUS=$SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd ~/Repos/Intern/residual-offpolicy-rl

# Short test: verify active_drawers + eval_active_drawers work correctly
# Train: D2×20, D3×5, D4×5 = 30 envs
# Eval:  D2×10, D3×5, D4×5 = 20 envs
# Only 5000 steps to check it runs without crashing

python3 resfit/rl_finetuning/scripts/train_residual_td3_unified.py \
    --config configs/tasks/drawer.json \
    --use_action_scaler \
    --no_action_clamp \
    --chunk_sync \
    --async_eval \
    --eval_interval 500 \
    --checkpoint_interval 1000 \
    --total_timesteps 1000 \
    --num_envs 30 --train_positions 30 \
    --active_drawers 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 \
    --eval_num_envs 20 --eval_positions 20 \
    --eval_active_drawers 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 \
    --wandb_mode disabled \
    --output_dir "outputs/drawer_dist_test_${SLURM_JOB_ID}" \
    "$@"
