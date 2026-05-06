#!/bin/bash
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=debug_vid
#SBATCH --output=/home/nas_main/kinamkim/slurms/debug_vid_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/debug_vid_%j.err
#SBATCH --time=00:30:00
#SBATCH --exclude=worker-6

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
python3 debug_video_collect.py
