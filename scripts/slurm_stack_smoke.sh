#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=60G
#SBATCH --job-name=stack_smoke
#SBATCH --output=/home/nas_main/kinamkim/slurms/stack_smoke_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/stack_smoke_%j.err
#SBATCH --nodelist=worker-6
#SBATCH --time=00:30:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_stack.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000 \
    --num_envs 4 \
    --total_timesteps 200 \
    --eval_interval 100 \
    --checkpoint_interval 9999 \
    --batch_size 64 \
    --buffer_size 5000 \
    --actor_lr 3e-4 --critic_lr 3e-4 \
    --gamma 0.95 \
    --target_tau 0.005 \
    --max_episode_steps 100 \
    --action_scale 0.1 \
    --random_action_noise_scale 0.05 \
    --action_l2_reg 1.0 \
    --critic_warmup_steps 50 \
    --n_step 3 \
    --seed 42 \
    --no_warmup \
    --output_dir outputs/stack_rl/smoke_test \
    --wandb_mode disabled \
    --residual_rot_scale 0.05 \
    --residual_pos_scale 0.02 \
    --residual_grip_scale 0.004 \
    --offline_fraction 0.0
