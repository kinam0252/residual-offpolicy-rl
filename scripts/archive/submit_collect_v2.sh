#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=col-v2
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/outputs/eval_logs/col_v2_%j.log
#SBATCH --error=/home/nas_main/kinamkim/Repos/Intern/outputs/eval_logs/col_v2_%j.log
#SBATCH --time=12:00:00

set -e
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

DIFFICULTY="${DIFFICULTY:-normal}"
GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"
EPISODE_POS="configs/pnp_eval_${DIFFICULTY}.json"
OUTPUT_DIR="outputs/offline_data/pnp_${DIFFICULTY}_v2_66ep100k"

echo "=== Offline Collection v2: ${DIFFICULTY} (66ep/100k, fixed eval positions) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Positions: $EPISODE_POS"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"
echo "============================================="

python3 resfit/rl_finetuning/scripts/collect_offline_data_pnp.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --episode_positions_file "$EPISODE_POS" \
    --num_episodes_per_env 34 \
    --max_episode_steps 500 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --ema_alpha 0.0 \
    --resume

echo "=== Done: $(date) ==="
