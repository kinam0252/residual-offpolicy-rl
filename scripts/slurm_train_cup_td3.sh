#!/bin/bash
#SBATCH --job-name=cup_rl
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/cup_rl_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/cup_rl_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/.local/lib:$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd ~/Repos/Intern/residual-offpolicy-rl

# ── Defaults (override via SBATCH env vars or edit below) ──
GROOT_CKPT=${GROOT_CKPT:-~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000}
OFFLINE_DIR=${OFFLINE_DIR:-outputs/offline_cup_batch}
NUM_ENVS=${NUM_ENVS:-27}
TOTAL_STEPS=${TOTAL_STEPS:-500000}
ACTION_SCALE=${ACTION_SCALE:-0.1}
ACTION_L2=${ACTION_L2:-1.0}
GAMMA=${GAMMA:-0.95}
OFFLINE_FRAC=${OFFLINE_FRAC:-0.75}
REWARD_TYPE=${REWARD_TYPE:-dense}
WANDB_NAME=${WANDB_NAME:-cup_rl}

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_cup.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --offline_data_dir "$OFFLINE_DIR" \
    --num_envs $NUM_ENVS \
    --total_timesteps $TOTAL_STEPS \
    --action_scale $ACTION_SCALE \
    --action_l2_reg $ACTION_L2 \
    --gamma $GAMMA \
    --offline_fraction $OFFLINE_FRAC \
    --reward_type $REWARD_TYPE \
    --use_action_scaler \
    --no_action_clamp \
    --async_eval \
    --eval_interval 2000 \
    --eval_num_episodes 1 \
    --eval_num_envs 20 \
    --checkpoint_interval 10000 \
    --wandb_mode online \
    --wandb_name "$WANDB_NAME" \
    --output_dir "outputs/cup_rl_${SLURM_JOB_ID}" \
    "$@"
