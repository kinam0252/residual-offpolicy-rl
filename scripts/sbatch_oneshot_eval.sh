#!/bin/bash
#SBATCH --job-name=oneshot-eval
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/oneshot_eval_%j.out

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/nas_main/.cache/huggingface DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
cd ~/Repos/Intern/residual-offpolicy-rl

EVAL_STDDEV=${EVAL_STDDEV:-0.0}
CKPT_DIR=${CKPT_DIR}
RESULT_DIR=${RESULT_DIR}

echo "[oneshot-eval] stddev=$EVAL_STDDEV ckpt_dir=$CKPT_DIR result_dir=$RESULT_DIR"

python resfit/rl_finetuning/scripts/eval_async_mujoco_drawer.py \
    --checkpoint_dir "$CKPT_DIR" \
    --output_dir "$RESULT_DIR" \
    --groot_checkpoint ~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000 \
    --groot_embodiment_tag new_embodiment \
    --task_description "Close the drawer" \
    --open_loop_horizon 1 \
    --residual_pos_scale 0.02 \
    --residual_rot_scale 0.05 \
    --residual_grip_scale 0.004 \
    --ema_alpha 0.0 \
    --action_scale 0.05 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 1024 \
    --max_episode_steps 500 \
    --eval_num_episodes 20 \
    --eval_stddev $EVAL_STDDEV \
    --active_drawers 2 \
    --contact_z_gate \
    --use_action_scaler \
    --action_scaler_min 0.240120 -0.264707 0.149373 0.000083 \
    --action_scaler_max 0.524598 0.178428 0.346476 0.000084 \
    --poll_interval_sec 1 \
    --wandb_mode disabled
