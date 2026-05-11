#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Object State Augmentation — Full Ablation Sweep (5 tasks × 4 conditions)
#
# Reproduces real-deploy training conditions + tests noise/dropout.
# Per-task overrides match REAL_DEPLOY_CHECKPOINTS.md settings.
#
# Usage:
#   bash scripts/slurm/sweep_obs_augment_all.sh              # all 5 tasks
#   TASKS="cup drawer" bash scripts/slurm/sweep_obs_augment_all.sh  # subset
#
# Defaults:  OBS_NOISE_MAX=0.01, OBS_DROPOUT_PROB=0.1
# Override:  OBS_NOISE_MAX=0.02 OBS_DROPOUT_PROB=0.15 bash scripts/slurm/sweep_obs_augment_all.sh
#
# DRY_RUN=1 bash scripts/slurm/sweep_obs_augment_all.sh      # preview only
# ══════════════════════════════════════════════════════════════════

set -e

NOISE=${OBS_NOISE_MAX:-0.01}
DROP=${OBS_DROPOUT_PROB:-0.1}
SCRIPT=scripts/slurm/train_td3.sh
DRY=${DRY_RUN:-0}
TASKS=${TASKS:-"cup lift pnp drawer stack"}

# ──────────────────────────────────────────────────────────────────
# Per-task overrides to match real-deploy training conditions
# Reference: report/deploy/REAL_DEPLOY_CHECKPOINTS.md
# ──────────────────────────────────────────────────────────────────
get_task_overrides() {
    local task=$1
    case $task in
        cup)
            # Deploy: action_scale=0.1, dense, gamma=0.95  → JSON matches
            echo ""
            ;;
        lift)
            # Deploy: action_scale=0.2 (v58_smallnet_old)
            # NOTE: deploy used clamp, but current code uses no_action_clamp (better)
            # Re-training from scratch with current arch, so use action_scale=0.2 only
            echo "ACTION_SCALE=0.2"
            ;;
        pnp)
            # Deploy: action_scale=0.1 (JSON default is 0.2)
            echo "ACTION_SCALE=0.1"
            ;;
        drawer)
            # Deploy: action_scale=0.05, reward_type=delta (JSON: 0.1, dense)
            echo "ACTION_SCALE=0.05 REWARD_TYPE=delta"
            ;;
        stack)
            # Deploy: action_scale=0.05 (JSON default is 0.1)
            echo "ACTION_SCALE=0.05"
            ;;
        *)
            echo "ERROR: Unknown task: $task" >&2
            exit 1
            ;;
    esac
}

# ──────────────────────────────────────────────────────────────────
# Submit one job
# ──────────────────────────────────────────────────────────────────
submit() {
    local task=$1 tag=$2 extra_env=$3
    local overrides
    overrides=$(get_task_overrides "$task")
    local job_name="${task}_aug_${tag}"
    local cmd="$overrides $extra_env TASK=$task WANDB_NAME=$job_name sbatch --job-name=$job_name $SCRIPT"

    if [ "$DRY" = "1" ]; then
        echo "  [DRY] $cmd"
    else
        echo "  → $job_name"
        eval "$cmd"
    fi
}

# ──────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Object State Augmentation — Full Ablation Sweep"
echo " noise_max=$NOISE | dropout_prob=$DROP"
echo " tasks: $TASKS"
[ "$DRY" = "1" ] && echo " *** DRY RUN — no jobs will be submitted ***"
echo "═══════════════════════════════════════════════════════════"
echo ""

total=0
for task in $TASKS; do
    echo "── $task ──────────────────────────────────────────────"
    overrides_desc=$(get_task_overrides "$task")
    [ -n "$overrides_desc" ] && echo "  overrides: $overrides_desc"

    # 1. clean (baseline)
    submit "$task" "clean" ""
    # 2. noise only
    submit "$task" "noise" "OBS_NOISE_MAX=$NOISE"
    # 3. dropout only
    submit "$task" "drop" "OBS_DROPOUT_PROB=$DROP"
    # 4. both
    submit "$task" "both" "OBS_NOISE_MAX=$NOISE OBS_DROPOUT_PROB=$DROP"

    total=$((total + 4))
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo " $total jobs submitted. Monitor: squeue -u \$(whoami)"
echo "═══════════════════════════════════════════════════════════"
