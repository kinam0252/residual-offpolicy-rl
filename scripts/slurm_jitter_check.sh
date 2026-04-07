#!/bin/bash
#SBATCH --job-name=jitter_check
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/jitter_check_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/jitter_check_%j.err

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
python3 resfit/rl_finetuning/scripts/check_chunk_jitter.py
