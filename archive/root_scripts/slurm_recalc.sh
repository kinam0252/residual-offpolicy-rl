#!/bin/bash
#SBATCH --job-name=recalc_reward
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/logs/recalc_reward_%j.log

source /home/nas_main/kinamkim/.venvs/groot/bin/activate
python -u /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/recalc_reward_resume.py
