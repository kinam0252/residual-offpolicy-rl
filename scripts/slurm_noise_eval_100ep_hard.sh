#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --job-name=noise_hard
#SBATCH --output=/home/nas_main/kinamkim/slurms/noise_hard_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/noise_hard_%j.err
#SBATCH --time=02:00:00

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
export PYTHONPATH=$PWD:$PYTHONPATH

echo '=== Noise: pos=0.005m rot=5deg ==='
python3 resfit/rl_finetuning/scripts/eval_checkpoint.py \
    --checkpoint /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/mujoco_td3/gr00t_sim_100ep/hard_s1L2_s2/checkpoints/best.pt \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000 \
    --perturb_table configs/eval_table_hard_v2.json \
    --eval_num_episodes 3 \
    --max_episode_steps 500 \
    --reward_type dense_clipped \
    --action_scale 1.0 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --asymmetric_critic \
    --calib_path /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml \
    --use_calibrated_wrist \
    --obj_pos_noise 0.005 \
    --obj_rot_noise 5

echo '=== Noise: pos=0.01m rot=10deg ==='
python3 resfit/rl_finetuning/scripts/eval_checkpoint.py \
    --checkpoint /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/mujoco_td3/gr00t_sim_100ep/hard_s1L2_s2/checkpoints/best.pt \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000 \
    --perturb_table configs/eval_table_hard_v2.json \
    --eval_num_episodes 3 \
    --max_episode_steps 500 \
    --reward_type dense_clipped \
    --action_scale 1.0 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --asymmetric_critic \
    --calib_path /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml \
    --use_calibrated_wrist \
    --obj_pos_noise 0.01 \
    --obj_rot_noise 10

echo '=== Noise: pos=0.02m rot=20deg ==='
python3 resfit/rl_finetuning/scripts/eval_checkpoint.py \
    --checkpoint /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/mujoco_td3/gr00t_sim_100ep/hard_s1L2_s2/checkpoints/best.pt \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000 \
    --perturb_table configs/eval_table_hard_v2.json \
    --eval_num_episodes 3 \
    --max_episode_steps 500 \
    --reward_type dense_clipped \
    --action_scale 1.0 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --asymmetric_critic \
    --calib_path /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml \
    --use_calibrated_wrist \
    --obj_pos_noise 0.02 \
    --obj_rot_noise 20

echo '=== Noise: pos=0.03m rot=30deg ==='
python3 resfit/rl_finetuning/scripts/eval_checkpoint.py \
    --checkpoint /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/mujoco_td3/gr00t_sim_100ep/hard_s1L2_s2/checkpoints/best.pt \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000 \
    --perturb_table configs/eval_table_hard_v2.json \
    --eval_num_episodes 3 \
    --max_episode_steps 500 \
    --reward_type dense_clipped \
    --action_scale 1.0 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --asymmetric_critic \
    --calib_path /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml \
    --use_calibrated_wrist \
    --obj_pos_noise 0.03 \
    --obj_rot_noise 30

echo '=== Noise: pos=0.05m rot=45deg ==='
python3 resfit/rl_finetuning/scripts/eval_checkpoint.py \
    --checkpoint /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/mujoco_td3/gr00t_sim_100ep/hard_s1L2_s2/checkpoints/best.pt \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000 \
    --perturb_table configs/eval_table_hard_v2.json \
    --eval_num_episodes 3 \
    --max_episode_steps 500 \
    --reward_type dense_clipped \
    --action_scale 1.0 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --asymmetric_critic \
    --calib_path /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml \
    --use_calibrated_wrist \
    --obj_pos_noise 0.05 \
    --obj_rot_noise 45

