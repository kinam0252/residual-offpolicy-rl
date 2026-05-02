#!/bin/bash
# SLURM submission for D2-only Drawer Residual TD3 training
#
# Usage:
#   AS=0.1 L2=0.0 CW=5000 SUFFIX=A sbatch scripts/slurm_train_drawer_td3_d2only.sh
#
# Parameterized via env vars:
#   AS       - action_scale (default 0.1)
#   L2       - action_l2_reg (default 0.0)
#   CW       - critic_warmup_steps (default 5000)
#   OF       - offline_ratio (default 0.5)
#   SUFFIX   - wandb run name suffix (default "")

#SBATCH --job-name=d2only-td3
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/d2only_td3_%j.out

set -e

# Environment setup
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

# Defaults
AS=${AS:-0.1}
L2=${L2:-0.0}
CW=${CW:-5000}
OF=${OF:-0.5}
SUFFIX=${SUFFIX:-""}

GROOT_CKPT=~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000
OFFLINE_DIR=outputs/offline_drawer_zgate_100k
OUTPUT_DIR=outputs/d2only_td3_as${AS}_L2${L2}_cw${CW}${SUFFIX}_$(date +%Y%m%d_%H%M%S)

WANDB_NAME="d2only_as${AS}_L2${L2}_cw${CW}${SUFFIX}"

python resfit/rl_finetuning/scripts/train_residual_td3_mujoco_drawer.py \
    --groot_checkpoint $GROOT_CKPT \
    --offline_data_dir $OFFLINE_DIR \
    --use_action_scaler \
    --action_scale $AS \
    --action_l2_reg $L2 \
    --active_drawers 2 \
    --contact_z_gate \
    --total_timesteps 50000 \
    --num_envs 30 \
    --async_eval \
    --eval_interval 2000 \
    --eval_num_episodes 20 \
    --offline_fraction $OF \
    --critic_warmup_steps $CW \
    --output_dir $OUTPUT_DIR \
    --wandb_project mujoco-drawer-residual-td3 \
    --wandb_name "$WANDB_NAME"
