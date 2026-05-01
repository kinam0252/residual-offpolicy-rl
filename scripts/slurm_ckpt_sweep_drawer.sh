#!/usr/bin/env bash
# Submit parallel close-drawer checkpoint sweep workers.
# Evaluates all checkpoints × 3 drawers × 10 trials = 330 work items.
# Usage: bash scripts/slurm_ckpt_sweep_drawer.sh [NUM_WORKERS]
set -euo pipefail

NUM_WORKERS=${1:-4}
REPO_DIR="$HOME/Repos/Intern/residual-offpolicy-rl"
OUT_DIR="outputs/ckpt_sweep_drawer"
LOG_DIR="$HOME/slurms"
mkdir -p "$LOG_DIR"

echo "Submitting $NUM_WORKERS ckpt sweep workers..."
echo "Work items: 330 (11 ckpts × 3 drawers × 10 trials)"

for i in $(seq 1 "$NUM_WORKERS"); do
    JOB_NAME="drw_cs_${i}"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition=core \
        --qos=core-extra \
        --gres=gpu:1 \
        --cpus-per-task=12 \
        --mem=120G \
        --time=08:00:00 \
        --output="${LOG_DIR}/ckpt_sweep_w${i}_%j.out" \
        --error="${LOG_DIR}/ckpt_sweep_w${i}_%j.err" \
        --wrap="
export LD_LIBRARY_PATH=\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
cd ${REPO_DIR}
\$HOME/.venvs/groot/bin/python resfit/rl_finetuning/scripts/eval_drawer_ckpt_sweep.py \
    --output_dir ${OUT_DIR}
"
    echo "  Submitted $JOB_NAME"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: find ${REPO_DIR}/${OUT_DIR} -name 'd*_t*.json' ! -name '*.tmp.json' | wc -l"
echo "Total work items: 330 (11 ckpts × 3 drawers × 10 trials)"
