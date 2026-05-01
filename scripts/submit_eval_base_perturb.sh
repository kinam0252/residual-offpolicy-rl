#!/bin/bash
#SBATCH --job-name=base_perturb
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/outputs/train_logs/slurm_%j.log
#SBATCH --error=/home/nas_main/kinamkim/Repos/Intern/outputs/train_logs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --time=02:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

N_ENVS=${N_ENVS:-50}
SEEDS=${SEEDS:-"0,1,2"}
LEVELS=${LEVELS:-"zero,easy,medium,hard,extreme"}
BATCH_SIZE=${BATCH_SIZE:-15}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/eval_base_perturb"}

echo "=== Base Policy Perturbation Eval ==="
echo "N_ENVS=$N_ENVS  SEEDS=$SEEDS  LEVELS=$LEVELS"
echo "BATCH_SIZE=$BATCH_SIZE  OUTPUT=$OUTPUT_DIR"
echo ""

python scripts/eval_base_perturb.py \
    --n-envs $N_ENVS \
    --batch-size $BATCH_SIZE \
    --seeds "$SEEDS" \
    --levels "$LEVELS" \
    --output-dir "$OUTPUT_DIR"
