#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=mj_s32_hard
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_s32_hard_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_s32_hard_%j.err
#SBATCH --time=72:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_32ep/checkpoint-300000 \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --reward_type dense \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr 3e-4 --critic_lr 3e-4 \
    --gamma 0.99 \
    --target_tau 0.005 \
    --eval_perturb_table configs/eval_table_hard_v2.json \
    --random_cube_range '{"dx":[-12,16],"dy":[-20,20],"yaw":[-30,30]}' \
    --num_envs 10 \
    --max_episode_steps 500 \
    --action_scale 0.2 \
    --random_action_noise_scale 0.05 \
    --action_l2_reg 0.1 \
    --offline_data_dir outputs/offline_data/gr00t_sim_32ep/hard_merged \
    --offline_fraction 0.2 \
    --critic_warmup_steps 2000 \
    --async_eval \
    --eval_num_episodes 2 \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --output_dir outputs/mujoco_td3/gr00t_sim_32ep/hard \
    --wandb_mode online \
    --wandb_project mujoco-resfit \
    --wandb_name sim32_hard
