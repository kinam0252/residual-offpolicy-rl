#!/bin/bash
#SBATCH --job-name=col100k_p6
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --exclude=worker-8
#SBATCH --output=/home/nas_main/kinamkim/slurms/col100k_p6_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/col100k_p6_%j.err
#SBATCH --time=4:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/collect_offline_data.py \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_sim_v2/checkpoint-100000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/perturb_table_sim_v2.json \
    --env_ids 12 13 \
    --num_episodes_per_env 10 \
    --max_episode_steps 500 \
    --output_dir outputs/offline_data/sim_groot_v2_100k_part6
