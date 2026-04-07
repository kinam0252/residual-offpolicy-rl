#!/bin/bash
#SBATCH --job-name=groot_ema_sweep
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/nas_main/kinamkim/slurms/groot_ema_sweep_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/groot_ema_sweep_%j.err
#SBATCH --time=02:00:00

source ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
python resfit/rl_finetuning/scripts/eval_groot_ema_sweep.py
