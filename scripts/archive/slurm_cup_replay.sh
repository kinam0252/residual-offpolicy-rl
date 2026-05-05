#!/bin/bash
#SBATCH --job-name=cup_replay
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/cup_replay_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/cup_replay_%j.err

set -euo pipefail

. ~/.venvs/groot/bin/activate

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$HOME/lib-compat:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

cd ~/Repos/Intern/residual-offpolicy-rl

python3 scripts/replay_cup_episode.py \
    --episode 0 \
    --output videos/cup_replay/ep000.mp4

echo "DONE"
