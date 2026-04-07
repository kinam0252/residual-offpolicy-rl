#!/bin/bash
#SBATCH --job-name=mj_eval_v87
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_eval_v87_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_eval_v87_%j.err

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/eval_random_video.py \
    --checkpoint outputs/mujoco_td3/v87_off02_tau005/checkpoints/best.pt \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --num_envs 10 \
    --num_episodes 5 \
    --max_episode_steps 500 \
    --reward_type dense \
    --action_scale 0.2 \
    --actor_hidden_dim 512 --critic_hidden_dim 1024 \
    --asymmetric_critic \
    --random_cube_range '{"dx":[-8,14],"dy":[-15,15],"yaw":[-10,10]}' \
    --video_dir outputs/eval_videos/v87_best_normal \
    --seed 42
