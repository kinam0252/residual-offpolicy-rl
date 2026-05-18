#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Stand Cup Robustness Ablation — Object State Augmentation (4 conditions)
#
# Base config: cup_v2 best (job 1961, peak 59.3%)
#   action_scale=0.3, dense_v3, gamma=0.95, offline=0.75, l2=1.0
#
# Conditions (clean already exists as job 1961):
#   noise   — obs_noise_max=0.01
#   drop    — obs_dropout_prob=0.1
#   both    — noise + dropout
#   no_obj  — object_state zeroed out entirely
#
# Usage:
#   bash scripts/slurm/sweep_cup_robustness.sh
#   DRY_RUN=1 bash scripts/slurm/sweep_cup_robustness.sh
#   CONDITIONS="noise both" bash scripts/slurm/sweep_cup_robustness.sh
# ══════════════════════════════════════════════════════════════════

set -e
cd ~/Repos/Intern/residual-offpolicy-rl

S=scripts/slurm/train_td3.sh
DRY=${DRY_RUN:-0}
CONDITIONS=${CONDITIONS:-"noise drop both no_obj"}

NOISE=0.01
DROP=0.1

# Cup v2 best config overrides (JSON defaults handle gamma/offline/l2)
BASE="TASK=cup REWARD_TYPE=dense_v3 ACTION_SCALE=0.3"

echo "═══════════════════════════════════════════════════════════"
echo " Stand Cup Robustness Ablation"
echo " conditions: $CONDITIONS"
echo " noise_max=$NOISE | dropout_prob=$DROP"
[ "$DRY" = "1" ] && echo " *** DRY RUN — no jobs will be submitted ***"
echo "═══════════════════════════════════════════════════════════"
echo ""

total=0
for cond in $CONDITIONS; do
    name="cup_aug_${cond}"
    case $cond in
        clean)  extra="" ;;
        noise)  extra="OBS_NOISE_MAX=$NOISE" ;;
        drop)   extra="OBS_DROPOUT_PROB=$DROP" ;;
        both)   extra="OBS_NOISE_MAX=$NOISE OBS_DROPOUT_PROB=$DROP" ;;
        no_obj) extra="DISABLE_OBJECT_STATE=1" ;;
        *)      echo "ERROR: Unknown condition: $cond"; exit 1 ;;
    esac

    cmd="$BASE $extra WANDB_NAME=$name sbatch --job-name=$name $S"

    if [ "$DRY" = "1" ]; then
        echo "  [DRY] $cmd"
    else
        echo "  → $name"
        eval "$cmd"
    fi
    total=$((total + 1))
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " $total jobs submitted. Monitor: squeue -u \$(whoami)"
echo "═══════════════════════════════════════════════════════════"
