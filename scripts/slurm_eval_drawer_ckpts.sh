#!/usr/bin/env bash
# Submit parallel close-drawer checkpoint eval workers.
# Usage: bash scripts/slurm_eval_drawer_ckpts.sh [NUM_WORKERS]
#   Default: 4 workers
set -euo pipefail

NUM_WORKERS=${1:-4}
REPO_DIR="$HOME/Repos/Intern/residual-offpolicy-rl"
OUT_DIR="outputs/ckpt_eval_drawer"
LOG_DIR="$HOME/slurms"
mkdir -p "$LOG_DIR"

echo "Submitting $NUM_WORKERS drawer eval workers..."
echo "Work items: 297 (9 ckpts × 33 episodes)"

for i in $(seq 1 "$NUM_WORKERS"); do
    JOB_NAME="drw_ev_${i}"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition=core \
        --gres=gpu:1 \
        --cpus-per-task=18 \
        --mem=200G \
        --time=06:00:00 \
        --output="${LOG_DIR}/drawer_eval_w${i}_%j.out" \
        --error="${LOG_DIR}/drawer_eval_w${i}_%j.err" \
        --wrap="
export LD_LIBRARY_PATH=\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
cd ${REPO_DIR}
\$HOME/.venvs/groot/bin/python resfit/rl_finetuning/scripts/eval_drawer_checkpoints.py \
    --output_dir ${OUT_DIR}
"
    echo "  Submitted $JOB_NAME"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ls ${REPO_DIR}/${OUT_DIR}/ckpt_*/ep*.json | wc -l"
echo "Total work items: 297 (9 ckpts × 33 episodes)"
