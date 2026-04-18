#!/bin/bash
#SBATCH --job-name=act_lift
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/act_lift_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/act_lift_%j.err

echo "=== Job Start: $(date), Node: $(hostname) ==="
nvidia-smi -L | head -2

# Prevent deepspeed from looking for nvcc at import time
export DS_BUILD_OPS=0

# torchcodec needs FFmpeg libs (from PyAV) + libtorch + CUDA NPP
TORCH_LIB=$(python3 -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
AV_LIBS=$(python3 -c "import av,os; print(os.path.join(os.path.dirname(os.path.dirname(av.__file__)),'av.libs'))")
NV_LIBS="/home/nas_main/kinamkim/.local/lib/python3.10/site-packages/nvidia/cu13/lib"
CUDA_LIBS="/home/nas_main/kinamkim/.local/lib/cuda_libs"
export LD_LIBRARY_PATH="$TORCH_LIB:$AV_LIBS:$NV_LIBS:$CUDA_LIBS:$LD_LIBRARY_PATH"

DATA_ROOT="/home/nas_main/kinamkim/Repos/Intern/.EXTERNAL/DATA/INTERN/Teleop/lerobot/Lift_224"
OUTPUT_DIR="/home/nas_main/kinamkim/Repos/Intern/.EXTERNAL/DATA/INTERN/Teleop/training/act_lift_224"
REPO_ID="kinamkim/fr3_lift_teleop_224"

# Clean previous output if exists (to avoid FileExistsError)
rm -rf "$OUTPUT_DIR"

lerobot-train \
    --policy.type=act \
    --policy.push_to_hub=false \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATA_ROOT" \
    --output_dir="$OUTPUT_DIR" \
    --steps=50000 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=0 \
    --batch_size=8 \
    --num_workers=8 \
    --wandb.enable=false

# Cleanup: keep only the last checkpoint + "last" symlink
echo "=== Cleaning up old checkpoints ==="
CKPT_DIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKPT_DIR" ]; then
    # Find the latest numeric checkpoint dir (excluding "last" symlink)
    LATEST=$(ls -d "$CKPT_DIR"/0* 2>/dev/null | sort | tail -1)
    for d in "$CKPT_DIR"/0*; do
        if [ "$d" != "$LATEST" ] && [ -d "$d" ]; then
            echo "  Removing old checkpoint: $d"
            rm -rf "$d"
        fi
    done
    echo "  Kept: $LATEST"
fi

echo "=== Job End: $(date) ==="
