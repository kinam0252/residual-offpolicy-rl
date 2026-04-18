#!/bin/bash
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=dbg_pnp_m
#SBATCH --output=/home/nas_main/kinamkim/slurms/dbg_pnp_m_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/dbg_pnp_m_%j.err
#SBATCH --time=1:00:00

. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
python3 scripts/debug_pnp_wrist.py
