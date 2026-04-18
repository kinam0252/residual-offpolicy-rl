#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=eval_pt_hard
#SBATCH --output=/home/nas_main/kinamkim/slurms/eval_pt_hard_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/eval_pt_hard_%j.err
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

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH=$PWD:$PYTHONPATH

python3 resfit/rl_finetuning/scripts/eval_all_envs_video.py \
    --groot_checkpoint nvidia/GR00T-N1.6-3B \
    --perturb_table configs/eval_table_hard_v2.json \
    --eval_num_episodes 1 \
    --max_episode_steps 500 \
    --reward_type dense_clipped \
    --action_scale 1.0 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --asymmetric_critic \
    --base_only \
    --camera back \
    --video_dir /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/eval_videos/pretrained_groot/hard \
    --seed 42
