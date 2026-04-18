#!/bin/bash
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=dbg_pnp
#SBATCH --output=/home/nas_main/kinamkim/slurms/dbg_pnp_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/dbg_pnp_%j.err
#SBATCH --time=1:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
python3 scripts/debug_pnp_video.py
