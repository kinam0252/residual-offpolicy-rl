#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --job-name=rl2_100ep_n_v2_noL2
#SBATCH --output=/home/nas_main/kinamkim/slurms/rl2_100ep_n_v2_noL2_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/rl2_100ep_n_v2_noL2_%j.err
#SBATCH --time=72:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export CUDA_HOME=~/fake_cuda
export PATH=~/bin:$PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY=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

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000 \
    --reward_type dense_clipped \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr 1e-5 --critic_lr 1e-4 \
    --gamma 0.99 \
    --target_tau 0.005 \
    --eval_perturb_table configs/eval_table_normal_v2.json \
    --random_cube_range '{"dx":[-8,10],"dy":[-12,12],"yaw":[0,0]}' \
    --num_envs 30 \
    --max_episode_steps 500 \
    --action_scale 1.0 \
    --random_action_noise_scale 0.05 \
    --action_l2_reg 0.0 \
    --offline_data_dir /home/nas_main/kinamkim/Repos/Intern/outputs/offline_data/gr00t_sim_100ep/normal \
    --offline_fraction 0.5 \
    --critic_warmup_steps 2000 \
    --async_eval \
    --eval_num_episodes 2 \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --n_step 3 \
    --seed 2 \
    --use_calibrated_wrist \
    --output_dir outputs/mujoco_td3/gr00t_sim_100ep/normal_v2_noL2 \
    --wandb_mode online \
    --wandb_project mujoco-resfit \
    --wandb_name rl2_100ep_normal_v2_noL2
