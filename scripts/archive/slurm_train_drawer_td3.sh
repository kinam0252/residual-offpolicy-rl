#!/bin/bash
# SLURM submission for Drawer Residual TD3 training
#
# Usage:
#   sbatch scripts/slurm_train_drawer_td3.sh
#
# Default: D2+D3+D4, 50K steps, ActionScaler, contact_z_gate, async eval

#SBATCH --job-name=drawer-td3
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/drawer_td3_%j.out

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

GROOT_CKPT=~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000
OFFLINE_DIR=outputs/offline_drawer_zgate_100k
OUTPUT_DIR=outputs/drawer_td3_$(date +%Y%m%d_%H%M%S)

mkdir -p slurm_logs

python resfit/rl_finetuning/scripts/train_residual_td3_mujoco_drawer.py \
    --groot_checkpoint $GROOT_CKPT \
    --offline_data_dir $OFFLINE_DIR \
    --use_action_scaler \
    --action_scale 0.1 \
    --action_l2_reg 0.0 \
    --active_drawers 2 3 4 \
    --contact_z_gate \
    --total_timesteps 50000 \
    --num_envs 30 \
    --async_eval \
    --eval_interval 2000 \
    --eval_num_episodes 5 \
    --output_dir $OUTPUT_DIR \
    --wandb_project mujoco-drawer-residual-td3 \
    --wandb_name "drawer_td3_D234_as01_noL2"
