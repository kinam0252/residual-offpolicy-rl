#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=col-v3
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/outputs/eval_logs/col_v3_%j.log
#SBATCH --error=/home/nas_main/kinamkim/Repos/Intern/outputs/eval_logs/col_v3_%j.log
#SBATCH --time=02:00:00

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
POS_FILE="configs/pnp_eval_${DIFFICULTY}.json"
OUTPUT_DIR="outputs/offline_data/pnp_${DIFFICULTY}_v3_66ep100k"

echo "=== Offline Collection v3 (batch): ${DIFFICULTY} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Positions: $POS_FILE"
echo "Output: $OUTPUT_DIR"
echo "Mode: 15-env batch inference"
echo "Start: $(date)"
echo "============================================="

python3 scripts/collect_pnp_batch_v3.py \
    --positions_file "$POS_FILE" \
    --groot_checkpoint "$GROOT_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --episodes_per_pos 34 \
    --max_episode_steps 500 \
    --ema_alpha 0.0 \
    --seed 42

echo "=== Done: $(date) ==="
