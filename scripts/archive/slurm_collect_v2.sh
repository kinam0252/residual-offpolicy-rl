#!/bin/bash
#SBATCH --job-name=collect_v2
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/collect_v2_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/collect_v2_%j.err

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
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/perturb_table_v2.json \
    --num_episodes_per_env 10 \
    --max_episode_steps 300 \
    --output_dir outputs/offline_data/dense_v2 \
    --device cuda:0

echo "DONE: offline data collection v2"
