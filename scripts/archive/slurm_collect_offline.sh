#!/bin/bash
#SBATCH --job-name=mj_offdata
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_offdata_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_offdata_%j.err

set -e
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Offline Data Collection ==="
echo "Job $SLURM_JOB_ID | $(hostname) | CUDA=$CUDA_VISIBLE_DEVICES"
echo "Episodes/env: 50 | Total: 500 episodes"

python3 resfit/rl_finetuning/scripts/collect_offline_data.py \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/mujoco_cube_perturb_table.json \
    --num_episodes_per_env 50 \
    --max_episode_steps 300 \
    --output_dir "outputs/offline_data/${TIMESTAMP}"
