#!/bin/bash
#SBATCH --job-name=col_100ep_e
#SBATCH --output=/home/nas_main/kinamkim/slurms/col_100ep_e_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/col_100ep_e_%j.out
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=06:00:00

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src:$PYTHONPATH

python3 resfit/rl_finetuning/scripts/collect_offline_data.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000 \
    --random_cube_range '{"dx":[-5,5],"dy":[-5,5]}' \
    --num_episodes_per_env 320 \
    --max_episode_steps 500 \
    --output_dir /home/nas_main/kinamkim/Repos/Intern/outputs/offline_data/gr00t_sim_100ep/easy \
    --use_calibrated_wrist \
    --calib_path /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml \
    --resume
