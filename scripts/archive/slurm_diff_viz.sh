#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --job-name=diff_viz
#SBATCH --output=/home/nas_main/kinamkim/slurms/diff_viz_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/diff_viz_%j.err
#SBATCH --time=00:10:00
#SBATCH --exclude=worker-6

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
python3 check_positions.py
echo "DONE"