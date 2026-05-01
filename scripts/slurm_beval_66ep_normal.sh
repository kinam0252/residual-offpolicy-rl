#!/bin/bash
#SBATCH --job-name=beval_66ep_n
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --output=/home/nas_main/kinamkim/slurms/beval_66ep_n_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/beval_66ep_n_%j.err
#SBATCH --time=4:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export EVAL_CKPT=/home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_66ep/checkpoint-300000
export EVAL_DIFF=normal
export EVAL_RCR='{"dx":[-8,14],"dy":[-15,15],"yaw":[-10,10]}'
export EVAL_OUT=outputs/66ep/base_eval_normal
export EVAL_SCENE=/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/../residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml
export EVAL_CALIB=/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
python3 resfit/rl_finetuning/scripts/eval_base_policy.py
