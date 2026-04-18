#!/bin/bash
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=chk_col
#SBATCH --output=/home/nas_main/kinamkim/slurms/chk_col_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/chk_col_%j.err
#SBATCH --time=0:10:00

. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
python3 scripts/check_collision2.py
