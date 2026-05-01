#!/usr/bin/env bash
# Submit parallel checkpoint eval workers.
# Usage: bash scripts/slurm_eval_stack_ckpts.sh [NUM_WORKERS]
#   Default: 7 workers on core-own
set -euo pipefail

NUM_WORKERS=${1:-7}
REPO_DIR="$HOME/Repos/Intern/residual-offpolicy-rl"
OUT_DIR="outputs/ckpt_eval_stack_66pos"
LOG_DIR="$HOME/slurms"
mkdir -p "$LOG_DIR"

echo "Submitting $NUM_WORKERS eval workers..."

for i in $(seq 1 "$NUM_WORKERS"); do
    JOB_NAME="stk_ev_${i}"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition=core \
        --gres=gpu:1 \
        --cpus-per-task=18 \
        --mem=200G \
        --time=06:00:00 \
        --output="${LOG_DIR}/stack_eval_w${i}_%j.out" \
        --error="${LOG_DIR}/stack_eval_w${i}_%j.err" \
        --wrap="
export LD_LIBRARY_PATH=\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
cd ${REPO_DIR}
\$HOME/.venvs/groot/bin/python resfit/rl_finetuning/scripts/eval_stack_checkpoints.py \
    --output_dir ${OUT_DIR}
"
    echo "  Submitted $JOB_NAME"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ls ${REPO_DIR}/${OUT_DIR}/ckpt_*/pos*.json | wc -l"
echo "Total work items: 792 (12 ckpts × 66 positions)"
