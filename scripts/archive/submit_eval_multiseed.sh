#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00

source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

CHECKPOINT="${CHECKPOINT:-}"
BASE_ONLY="${BASE_ONLY:-false}"
SEEDS="${SEEDS:-0_1_2}"
SEEDS=$(echo "$SEEDS" | tr '_' ',')

if [ "$BASE_ONLY" = "true" ]; then
    python3 scripts/eval_multiseed.py \
        --base-only \
        --positions-file configs/pnp_eval_hard.json \
        --seeds "$SEEDS"
else
    python3 scripts/eval_multiseed.py \
        --checkpoint "$CHECKPOINT" \
        --positions-file configs/pnp_eval_hard.json \
        --seeds "$SEEDS"
fi
