#!/bin/bash
#SBATCH --partition=free
#SBATCH --qos=core-on-free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=mj_d08L2h_s2
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_d08L2h_s2_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_d08L2h_s2_%j.err
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

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_32ep/checkpoint-300000 \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --reward_type dense_clipped \
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
    --num_envs 30 \
    --max_episode_steps 500 \
    --action_scale 0.8 \
    --random_action_noise_scale 0.05 \
    --action_l2_reg 0.05 \
    --offline_data_dir outputs/offline_data/gr00t_sim_32ep/hard_merged \
    --offline_fraction 0.5 \
    --critic_warmup_steps 2000 \
    --async_eval \
    --eval_num_episodes 2 \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --output_dir outputs/mujoco_td3/gr00t_sim_32ep/denseClip_s08_L2_hard_s2 \
    --wandb_mode online \
    --wandb_project mujoco-resfit \
    --wandb_name sim32_denseClip_s08_L2_hard_s2
