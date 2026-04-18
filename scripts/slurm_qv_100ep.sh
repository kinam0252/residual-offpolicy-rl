#!/bin/bash
#SBATCH --job-name=qv_100ep
#SBATCH --output=/home/nas_main/kinamkim/slurms/qv_100ep_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/qv_100ep_%j.out
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=00:30:00

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src:$PYTHONPATH
export EVAL_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000"
export EVAL_NAME="100ep"
export EVAL_CALIB="/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml"
export EVAL_OUT="/home/nas_main/kinamkim/Repos/Intern/outputs/100ep/base_eval_video"

python3 /home/nas_main/kinamkim/Repos/Intern/scripts/eval_base_video.py
