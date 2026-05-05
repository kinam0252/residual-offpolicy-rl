#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=col-v3p
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/outputs/eval_logs/col_par_%j.log
#SBATCH --error=/home/nas_main/kinamkim/Repos/Intern/outputs/eval_logs/col_par_%j.log
#SBATCH --time=02:00:00

set -e
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"
POS_FILE="configs/pnp_66ep_positions.json"
OUTPUT_DIR="outputs/offline_data/pnp_66ep_dense_v3"
REWARD_TYPE="dense_v3"
EP_PER_POS="${EP_PER_POS:-10}"

echo "=== Parallel Offline Collection ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Positions: $POS_FILE"
echo "Output: $OUTPUT_DIR"
echo "Reward: $REWARD_TYPE"
echo "Eps/pos: $EP_PER_POS"
echo "Start: $(date)"
echo "===================================="

python3 scripts/collect_pnp_parallel.py \
    --positions_file "$POS_FILE" \
    --groot_checkpoint "$GROOT_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --reward_type "$REWARD_TYPE" \
    --episodes_per_pos "$EP_PER_POS" \
    --batch_size 15 \
    --max_episode_steps 500 \
    --seed $RANDOM

echo "=== Done: $(date) ==="
