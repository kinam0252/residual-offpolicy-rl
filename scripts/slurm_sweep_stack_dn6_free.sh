#!/bin/bash
#SBATCH --job-name=stk_dn6
#SBATCH --partition=core
#SBATCH --qos=core-on-free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/stk_dn6_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/stk_dn6_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DS_BUILD_OPS=0
export CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd ~/Repos/Intern/residual-offpolicy-rl

# Exp6: dense control + action_scale=0.05 (reduce saturation baseline)
python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_stack.py \
    --groot_checkpoint ~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000 \
    --use_action_scaler \
    --no_action_clamp \
    --eval_num_envs 20 \
    --offline_data_dir outputs/offline_stack_66ep \
    --offline_fraction 0.75 \
    --num_envs 30 \
    --episode_positions_file configs/stack_cube_positions.json \
    --eval_positions_file configs/stack_cube_positions.json \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr 3e-4 --critic_lr 3e-4 \
    --gamma 0.95 \
    --target_tau 0.005 \
    --max_episode_steps 300 \
    --random_action_noise_scale 0.05 \
    --action_l2_reg 1.0 \
    --critic_warmup_steps 2000 \
    --n_step 3 \
    --async_eval \
    --no_offline_cache \
    --eval_num_episodes 5 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --wandb_mode offline \
    --reward_type dense \
    --offline_reward_relabel none \
    --residual_rot_scale 0.05 \
    --action_scale 0.05 \
    --seed 42 \
    --output_dir outputs/stack_rl/dn6_g95_n3_a05
