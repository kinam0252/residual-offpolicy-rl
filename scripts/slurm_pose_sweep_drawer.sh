#!/usr/bin/env bash
# Submit parallel close-drawer pose sweep workers.
# Usage: bash scripts/slurm_pose_sweep_drawer.sh [NUM_WORKERS]
#   Default: 4 workers (core-extra QoS)
set -euo pipefail

NUM_WORKERS=${1:-4}
REPO_DIR="$HOME/Repos/Intern/residual-offpolicy-rl"
OUT_DIR="outputs/pose_sweep_drawer"
LOG_DIR="$HOME/slurms"
mkdir -p "$LOG_DIR"

echo "Submitting $NUM_WORKERS pose sweep workers..."
echo "Work items: 525 (175 poses × 3 drawers)"

for i in $(seq 1 "$NUM_WORKERS"); do
    JOB_NAME="drw_ps_${i}"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition=core \
        --qos=core-extra \
        --gres=gpu:1 \
        --cpus-per-task=12 \
        --mem=120G \
        --time=08:00:00 \
        --output="${LOG_DIR}/pose_sweep_w${i}_%j.out" \
        --error="${LOG_DIR}/pose_sweep_w${i}_%j.err" \
        --wrap="
export LD_LIBRARY_PATH=\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
cd ${REPO_DIR}
\$HOME/.venvs/groot/bin/python resfit/rl_finetuning/scripts/eval_drawer_pose_sweep.py \
    --output_dir ${OUT_DIR}
"
    echo "  Submitted $JOB_NAME"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: find ${REPO_DIR}/${OUT_DIR} -name 'd*.json' ! -name '*.tmp.json' | wc -l"
echo "Total work items: 525 (175 poses × 3 drawers)"
