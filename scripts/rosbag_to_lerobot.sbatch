#!/bin/bash
#SBATCH --job-name=rosbag2lerobot
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/scripts/slurm_rosbag2lerobot_%j.out
#SBATCH --error=/home/nas_main/kinamkim/Repos/Intern/scripts/slurm_rosbag2lerobot_%j.err

echo "=== Job Start: $(date), Node: $(hostname), CPUs: $SLURM_CPUS_PER_TASK ==="

INPUT_DIR="/home/nas_main/kinamkim/Repos/Intern/.EXTERNAL/DATA/INTERN/Teleop/original/Lift"
OUTPUT_DIR="/home/nas_main/kinamkim/Repos/Intern/.EXTERNAL/DATA/INTERN/Teleop/lerobot/Lift"

python3 /home/nas_main/kinamkim/Repos/Intern/scripts/rosbag_to_lerobot.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --repo-id "kinamkim/fr3_lift_teleop" \
    --fps 15 \
    --task "lift" \
    --image-writer-threads 8

echo "=== Job End: $(date) ==="
