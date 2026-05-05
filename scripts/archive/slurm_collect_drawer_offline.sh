#!/bin/bash
# Submit multiple workers for drawer offline collection.
# Usage: bash scripts/slurm_collect_drawer_offline.sh [NUM_WORKERS] [PARTITION] [QOS]

NUM_WORKERS=${1:-7}
PARTITION=${2:-free}
QOS=${3:-core-on-free}

CKPT="$HOME/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000"
OUTPUT="outputs/offline_drawer_zgate_100k"

echo "=== Drawer Offline Collection (contact_z_gate) ==="
echo "  Workers: $NUM_WORKERS"
echo "  Partition: $PARTITION, QOS: $QOS"
echo "  Checkpoint: $CKPT"
echo "  Output: $OUTPUT"
echo ""

for i in $(seq 1 $NUM_WORKERS); do
    JOB_NAME="drawer_offline_w${i}"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --qos="$QOS" \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        --wrap="#!/bin/bash
. \$HOME/.venvs/groot/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/nas_main/.cache/huggingface
export DS_BUILD_OPS=0 CUDA_HOME=\$HOME/fake_cuda
NV=\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=\$HOME/lib-compat:\$NV/cu13/lib:\$NV/cuda_runtime/lib:\$NV/cublas/lib:\$NV/cudnn/lib:\$NV/cufft/lib:\$NV/cusolver/lib:\$NV/cusparse/lib:\$NV/nvjitlink/lib:\$NV/cuda_nvrtc/lib:\$NV/nccl/lib:\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
cd $HOME/Repos/Intern/residual-offpolicy-rl
python scripts/collect_drawer_offline.py \
    --groot_checkpoint $CKPT \
    --active_drawers 2 3 4 \
    --num_episodes 30 \
    --output_dir $OUTPUT \
    --contact_z_gate
"
    echo "  Submitted worker $i: $JOB_NAME"
done

echo ""
echo "Done. Monitor with: ls $OUTPUT/*.npz | wc -l"
