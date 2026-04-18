#!/bin/bash
#SBATCH --job-name=pnp-eval-sweep
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/pnp_eval_sweep/slurm_%j.log
#SBATCH --error=outputs/pnp_eval_sweep/slurm_%j.log

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "================"

cd ~/Repos/Intern
mkdir -p outputs/pnp_eval_sweep

source ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}

python scripts/eval_pnp_sweep.py \
    --output-dir outputs/pnp_eval_sweep

echo "=== Done: $(date) ==="
