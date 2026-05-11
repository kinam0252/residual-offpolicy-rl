#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Robustness Eval Sweep — 8 checkpoints × (clean + noisy)
#
# Submits 8 SLURM jobs (1 per checkpoint), each runs clean + noisy eval.
# Uses core-own QoS (8 jobs fit exactly).
#
# Usage:
#   bash scripts/slurm/sweep_robustness_eval.sh          # submit all
#   DRY_RUN=1 bash scripts/slurm/sweep_robustness_eval.sh  # preview
#
# Defaults: noise=0.01, dropout=0.1, eval_episodes=3
# Override: NOISE_MAX=0.02 DROPOUT_PROB=0.15 EVAL_EPISODES=5 bash ...
# ══════════════════════════════════════════════════════════════════

set -e

NOISE=${NOISE_MAX:-0.01}
DROP=${DROPOUT_PROB:-0.1}
EPISODES=${EVAL_EPISODES:-3}
DRY=${DRY_RUN:-0}
SCRIPT=scripts/slurm/eval_robustness.sh

echo "═══════════════════════════════════════════════════════"
echo " Robustness Eval Sweep"
echo " noise_max=$NOISE | dropout_prob=$DROP | episodes=$EPISODES"
[ "$DRY" = "1" ] && echo " *** DRY RUN ***"
echo "═══════════════════════════════════════════════════════"
echo ""

total=0

submit() {
    local task=$1 condition=$2 ckpt_dir=$3
    local job_name="rob_${task}_${condition}"
    local cmd="TASK=$task CKPT_DIR=$ckpt_dir CONDITION=$condition \
NOISE_MAX=$NOISE DROPOUT_PROB=$DROP EVAL_EPISODES=$EPISODES \
sbatch --job-name=$job_name $SCRIPT"

    if [ "$DRY" = "1" ]; then
        echo "  [DRY] $cmd"
    else
        echo "  → $job_name ($ckpt_dir)"
        eval "$cmd"
    fi
    total=$((total + 1))
}

# ── PNP checkpoints ──
echo "── PNP ──"
submit pnp clean  outputs/pnp_rl_747
submit pnp noise  outputs/pnp_rl_748
submit pnp drop   outputs/pnp_rl_749
submit pnp both   outputs/pnp_rl_750
echo ""

# ── Stack checkpoints ──
echo "── Stack ──"
submit stack clean  outputs/stack_rl_763
submit stack noise  outputs/stack_rl_764
submit stack drop   outputs/stack_rl_765
submit stack both   outputs/stack_rl_766
echo ""

echo "═══════════════════════════════════════════════════════"
echo " $total jobs submitted. Monitor: squeue -u \$(whoami)"
echo "═══════════════════════════════════════════════════════"
