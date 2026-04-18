#!/bin/bash
# Submit N parallel eval workers for PnP base policy evaluation.
# Each worker independently picks up remaining combos (work-stealing).
# Usage: bash scripts/submit_pnp_eval_all.sh [NUM_WORKERS]

NUM_WORKERS=${1:-8}

echo "Submitting $NUM_WORKERS PnP eval workers..."

for i in $(seq 1 $NUM_WORKERS); do
    JOB_ID=$(sbatch --parsable scripts/slurm_eval_pnp_base.sh)
    echo "  Worker $i: job $JOB_ID"
done

echo "All $NUM_WORKERS workers submitted. They will self-schedule across 27 combos."
echo "Monitor: squeue -u \$USER | grep pnp_eval"
echo "Results: ls outputs/pnp_base_eval/*/results.json | wc -l"
