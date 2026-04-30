#!/bin/bash
# Submit video rollout jobs for trajectory analysis pairs.
# Positions where residual > base:
#   Stack: 0, 5, 6, 8, 14, 15
#   PnP: 1
# Also include "both succeed but residual faster" positions for efficiency comparison.

set -e
cd "$(dirname "$0")/.."

TEMPLATE=~/Reports/slurm/template.sh
OUT_DIR="outputs/trajectory_videos"
mkdir -p slurms

# Jobs:
# 1. stack_base (positions 0,5,6,8,14,15) - 3 rounds
# 2. stack_residual (positions 0,5,6,8,14,15) - 3 rounds
# 3. pnp_base (position 1) - 3 rounds
# 4. pnp_residual (position 1) - 3 rounds
# 5. stack_base "efficiency" (positions where both succeed: 14,15,3,4,7) - 3 rounds
# 6. stack_residual "efficiency" (same) - 3 rounds

submit_job() {
    local name=$1
    local cmd=$2
    echo "Submitting: $name"
    sbatch --job-name="vid_${name}" \
           --partition=core \
           --qos=core-own \
           --gres=gpu:1 \
           --cpus-per-task=14 \
           --mem=200G \
           --time=2:00:00 \
           --output="slurms/vid_${name}_%j.out" \
           --comment=pytorch \
           "$TEMPLATE" "$cmd"
}

# Activate env command prefix (CUDA 13.0 compatible)
NV="\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia"
NV_LD="${NV}/cu13/lib:${NV}/cuda_runtime/lib:${NV}/cublas/lib:${NV}/cudnn/lib:${NV}/cufft/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${NV}/cuda_nvrtc/lib:${NV}/nccl/lib"
ENV_CMD=". ~/.venvs/groot/bin/activate && export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,) && export LD_LIBRARY_PATH=${NV_LD}:\$HOME/lib-compat:~/.local/lib/gl:\${LD_LIBRARY_PATH:-} && export MUJOCO_GL=egl DS_BUILD_OPS=0 HF_HUB_OFFLINE=1 CUDA_HOME=~/fake_cuda && export PYTHONPATH=$(pwd):\${PYTHONPATH:-} && cd $(pwd)"

# 1. Stack base - residual-better positions
submit_job "stack_base_improve" \
    "${ENV_CMD} && python scripts/rollout_video.py --task stack --mode base --positions 0,5,6,8,14,15 --num_rounds 3 --output_dir ${OUT_DIR}"

# 2. Stack residual - residual-better positions
submit_job "stack_res_improve" \
    "${ENV_CMD} && python scripts/rollout_video.py --task stack --mode residual --positions 0,5,6,8,14,15 --num_rounds 3 --output_dir ${OUT_DIR}"

# 3. PnP base - position 1 (base fails, residual succeeds)
submit_job "pnp_base_pos1" \
    "${ENV_CMD} && python scripts/rollout_video.py --task pnp --mode base --positions 1 --num_rounds 3 --output_dir ${OUT_DIR}"

# 4. PnP residual - position 1
submit_job "pnp_res_pos1" \
    "${ENV_CMD} && python scripts/rollout_video.py --task pnp --mode residual --positions 1 --num_rounds 3 --output_dir ${OUT_DIR}"

# 5. Stack base - efficiency positions (both succeed, residual faster)
submit_job "stack_base_eff" \
    "${ENV_CMD} && python scripts/rollout_video.py --task stack --mode base --positions 3,4,7,14,15 --num_rounds 3 --output_dir ${OUT_DIR}"

# 6. Stack residual - efficiency positions
submit_job "stack_res_eff" \
    "${ENV_CMD} && python scripts/rollout_video.py --task stack --mode residual --positions 3,4,7,14,15 --num_rounds 3 --output_dir ${OUT_DIR}"

echo ""
echo "All 6 video jobs submitted. Check with: squeue -u \$USER"
