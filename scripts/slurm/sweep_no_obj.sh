#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# No Object State Ablation — Train without object_state observation
#
# Tests whether the RL agent can learn using only robot state (10D)
# + GR00T base_action (7D), without any object pose information.
#
# Uses same per-task configs as the original ablation sweep.
#
# Usage:
#   bash scripts/slurm/sweep_no_obj.sh              # all 3 tasks
#   TASKS="pnp stack" bash scripts/slurm/sweep_no_obj.sh  # subset
#   DRY_RUN=1 bash scripts/slurm/sweep_no_obj.sh    # preview only
# ══════════════════════════════════════════════════════════════════

set -e

SCRIPT=scripts/slurm/train_td3.sh
DRY=${DRY_RUN:-0}
TASKS=${TASKS:-"pnp stack drawer"}

# Per-task overrides (same as original ablation sweep)
get_task_overrides() {
    local task=$1
    case $task in
        pnp)    echo "ACTION_SCALE=0.1" ;;
        stack)  echo "ACTION_SCALE=0.05" ;;
        drawer) echo "ACTION_SCALE=0.05 REWARD_TYPE=delta" ;;
        *)      echo "ERROR: Unknown task: $task" >&2; exit 1 ;;
    esac
}

echo "═══════════════════════════════════════════════════════════"
echo " No Object State Ablation"
echo " tasks: $TASKS"
[ "$DRY" = "1" ] && echo " *** DRY RUN — no jobs will be submitted ***"
echo "═══════════════════════════════════════════════════════════"
echo ""

total=0
for task in $TASKS; do
    overrides=$(get_task_overrides "$task")
    job_name="${task}_aug_no_obj"
    cmd="$overrides DISABLE_OBJECT_STATE=1 TASK=$task WANDB_NAME=$job_name sbatch --job-name=$job_name $SCRIPT"

    if [ "$DRY" = "1" ]; then
        echo "  [DRY] $cmd"
    else
        echo "  → $job_name"
        eval "$cmd"
    fi
    total=$((total + 1))
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " $total jobs submitted. Monitor: squeue -u \$(whoami)"
echo "═══════════════════════════════════════════════════════════"
