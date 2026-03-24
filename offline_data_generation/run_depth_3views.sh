#!/bin/bash
#SBATCH --job-name=depth_3v
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=24
#SBATCH --output=/home/nas_main/kinamkim/slurms/depth_3views_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/depth_3views_%j.err
#SBATCH --time=0-02:00:00

# ============================================================
# Depth comparison: 3 views (front/back/wrist) × DA-V2 vs GT
# Isaac Sim replay + Depth-Anything-V2 inference
# ============================================================

source /home/nas_main/kinamkim/Repos/Intern/Honda_IsaacLab/apptainer/env.sh

ASSETS=/home/nas_main/kinamkim/Repos/Intern/assets
CSV_BASE_DIR=${ASSETS}/csv_data/pickMushroom_20251209_081744_174
RESRL=${RESRL_PATH}
DA_ENCODER="vitl"
EPISODE_IDX=0

OUTPUT_DIR=/home/nas_main/kinamkim/Repos/Intern/FoundationPose/eval_output/depth_3views
mkdir -p ${OUTPUT_DIR}
mkdir -p /home/nas_main/kinamkim/slurms

echo "=== Depth 3-View Comparison ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Episode: ${EPISODE_IDX}"
echo "Output: ${OUTPUT_DIR}"
echo ""

isaac_python \
    ${RESRL}/offline_data_generation/replay_depth_all_views.py \
    --headless \
    --enable_cameras \
    --num_envs 1 \
    --csv_base_dir "${CSV_BASE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --episode_idx ${EPISODE_IDX} \
    --da_encoder ${DA_ENCODER}

echo ""
echo "=== Done ==="
echo "Output files:"
ls -lh ${OUTPUT_DIR}/episode_*.mp4 2>/dev/null
