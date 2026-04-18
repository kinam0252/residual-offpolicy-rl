#!/bin/bash
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=pnp_infer_66ep
#SBATCH --output=/home/nas_main/kinamkim/slurms/pnp_infer_66ep_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/pnp_infer_66ep_%j.err
#SBATCH --time=2:00:00

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
python3 scripts/debug_pnp_infer.py
