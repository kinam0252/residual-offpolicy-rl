#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --job-name=evalall_100ep_hard
#SBATCH --output=/home/nas_main/kinamkim/slurms/evalall_100ep_hard_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/evalall_100ep_hard_%j.err
#SBATCH --time=01:00:00

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

python3 resfit/rl_finetuning/scripts/eval_all_envs_video.py \
    --checkpoint /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/mujoco_td3/gr00t_sim_100ep/hard_s1L2_s2/checkpoints/best.pt \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000 \
    --perturb_table configs/eval_table_hard_v2.json \
    --eval_num_episodes 1 \
    --max_episode_steps 500 \
    --reward_type dense_clipped \
    --action_scale 1.0 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --asymmetric_critic \
    --calib_path /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml \
    --use_calibrated_wrist \
    --camera back \
    --video_dir /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/eval_videos/100ep_allenv/hard \
    --seed 42
