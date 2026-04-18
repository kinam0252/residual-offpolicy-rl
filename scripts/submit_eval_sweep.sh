#!/bin/bash
#SBATCH --job-name=pnp-eval
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/outputs/eval_logs/slurm_%j.log
#SBATCH --error=/home/nas_main/kinamkim/Repos/Intern/outputs/eval_logs/slurm_%j.log

DIFFICULTY=${1:-easy}

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Difficulty: $DIFFICULTY"
echo "Start: $(date)"
echo "================"

OUTDIR=/home/nas_main/kinamkim/Repos/Intern/outputs/pnp_eval_sweep_${DIFFICULTY}
mkdir -p $OUTDIR

cd ~/Repos/Intern/residual-offpolicy-rl

source ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}

python scripts/eval_pnp_sweep.py \
    --difficulty $DIFFICULTY \
    --output-dir $OUTDIR

echo "=== Done: $(date) ==="
