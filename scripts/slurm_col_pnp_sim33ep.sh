#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=col_pnp_33
#SBATCH --output=/home/nas_main/kinamkim/slurms/col_pnp_33_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/col_pnp_33_%j.err
#SBATCH --time=24:00:00

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

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/collect_offline_data_pnp.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-75000 \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --cube_pos 0.42 -0.03 0.02 \
    --bowl_pos 0.42 0.03 0.0 \
    --num_episodes_per_env 500 \
    --max_episode_steps 500 \
    --task_description "Pick up the red cube and place it onto the plate." \
    --output_dir outputs/offline_data/pnp_sim33ep
