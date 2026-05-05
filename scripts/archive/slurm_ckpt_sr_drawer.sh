#!/usr/bin/env bash
# Submit parallel drawer checkpoint SR eval workers.
# Usage: bash scripts/slurm_ckpt_sr_drawer.sh [NUM_WORKERS] [QOS]
set -euo pipefail

NUM_WORKERS=${1:-4}
QOS=${2:-core-on-free}
REPO_DIR="$HOME/Repos/Intern/residual-offpolicy-rl"
OUT_DIR="outputs/ckpt_sr_drawer"
LOG_DIR="$HOME/slurms"
mkdir -p "$LOG_DIR" "${REPO_DIR}/${OUT_DIR}"

echo "Submitting $NUM_WORKERS drawer SR eval workers (QOS=$QOS)..."
echo "Work items: 36 (12 ckpts × 3 drawers, 20 episodes each)"

for i in $(seq 1 "$NUM_WORKERS"); do
    JOB_NAME="drw_sr_${i}"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition=free \
        --qos="$QOS" \
        --gres=gpu:1 \
        --cpus-per-task=12 \
        --mem=120G \
        --time=04:00:00 \
        --output="${LOG_DIR}/drawer_sr_w${i}_%j.out" \
        --error="${LOG_DIR}/drawer_sr_w${i}_%j.err" \
        --wrap="#!/bin/bash
. \$HOME/.venvs/groot/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/nas_main/.cache/huggingface
export DS_BUILD_OPS=0 CUDA_HOME=\$HOME/fake_cuda
NV=\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=\$HOME/lib-compat:\$NV/cu13/lib:\$NV/cuda_runtime/lib:\$NV/cublas/lib:\$NV/cudnn/lib:\$NV/cufft/lib:\$NV/cusolver/lib:\$NV/cusparse/lib:\$NV/nvjitlink/lib:\$NV/cuda_nvrtc/lib:\$NV/nccl/lib:\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
cd ${REPO_DIR}
python scripts/eval_drawer_ckpt_sr.py --output_dir ${OUT_DIR} --num_episodes 20 --max_steps 300
"
    echo "  Submitted $JOB_NAME"
done

echo ""
echo "Monitor: squeue -u \$USER | grep drw_sr"
echo "Progress: ls ${REPO_DIR}/${OUT_DIR}/*.json 2>/dev/null | wc -l"
echo "Total: 36 items"
