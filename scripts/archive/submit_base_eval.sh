#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=base_eval
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/outputs/train_logs/slurm_%j.log
#SBATCH --error=/home/nas_main/kinamkim/Repos/Intern/outputs/train_logs/slurm_%j.err
#SBATCH --time=2:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

POSITIONS_FILE="${POSITIONS_FILE:?Must set POSITIONS_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:?Must set OUTPUT_DIR}"
N_ENVS="${N_ENVS:-15}"

echo "[base-eval] positions=$POSITIONS_FILE n_envs=$N_ENVS output=$OUTPUT_DIR"
echo "[base-eval] Start: $(date)"

python3 scripts/debug_reward_v3.py \
    --positions-file "$POSITIONS_FILE" \
    --n-envs "$N_ENVS" \
    --output-dir "$OUTPUT_DIR"

echo "[base-eval] Done: $(date)"
