#!/bin/bash
#SBATCH --job-name=collect_rand_normal
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/collect_rand_normal_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/collect_rand_normal_%j.err

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
    --random_cube_range '{"dx":[-8,14],"dy":[-15,15],"yaw":[-10,10]}' \
    --num_episodes_per_env 100 \
    --max_episode_steps 300 \
    --output_dir outputs/offline_data/dense_rand_normal \
    --device cuda:0
