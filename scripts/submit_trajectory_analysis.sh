#!/usr/bin/env bash
# Submit trajectory analysis rollouts for 3 tasks × 2 modes = 6 SLURM jobs.
# Usage: bash scripts/submit_trajectory_analysis.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/rollout_trajectory_analysis.py"
OUTPUT_DIR="$REPO_DIR/outputs/trajectory_analysis"
mkdir -p "$OUTPUT_DIR"

PARTITION="core"
QOS="core-own"
GPUS=1
CPUS=14
MEM="200G"
TIME="04:00:00"
SLURM_OUT_DIR="$HOME/slurms"
mkdir -p "$SLURM_OUT_DIR"

for TASK in lift pnp stack; do
  for MODE in base residual; do
    JOB_NAME="traj_${TASK}_${MODE}"
    SLURM_LOG="${SLURM_OUT_DIR}/${JOB_NAME}_%j"

    sbatch --job-name="$JOB_NAME" \
           --partition="$PARTITION" \
           --qos="$QOS" \
           --gres=gpu:$GPUS \
           --cpus-per-task=$CPUS \
           --mem=$MEM \
           --time=$TIME \
           --output="${SLURM_LOG}.out" \
           --error="${SLURM_LOG}.err" \
           --wrap="#!/bin/bash
. ~/.venvs/groot/bin/activate
export LD_LIBRARY_PATH=~/.local/lib/gl:\${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export CUDA_HOME=~/fake_cuda
cd $REPO_DIR
export PYTHONPATH=\$(pwd):\${PYTHONPATH:-}

python $SCRIPT \\
  --task $TASK \\
  --mode $MODE \\
  --num_episodes 1 \\
  --output_dir $OUTPUT_DIR \\
  --device cuda
"
    echo "Submitted: $JOB_NAME"
    sleep 2
  done
done

echo ""
echo "All 6 jobs submitted. Results will be saved to: $OUTPUT_DIR"
echo "Monitor: squeue -u \$USER"
