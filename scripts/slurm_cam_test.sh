#!/bin/bash
#SBATCH --job-name=cam_test
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/nas_main/kinamkim/slurms/cam_test_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/cam_test_%j.err
#SBATCH --time=00:30:00

source ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
python resfit/rl_finetuning/scripts/cam_test.py
