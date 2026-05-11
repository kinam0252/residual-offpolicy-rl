#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Object State Augmentation Ablation Sweep
#
# Submits 4 jobs for each combination of noise/dropout:
#   1. both   (noise + dropout)   — full augmentation
#   2. noise  (noise only)        — noise effect isolation
#   3. drop   (dropout only)      — dropout effect isolation
#   4. clean  (no augmentation)   — baseline
#
# Usage:
#   TASK=cup bash scripts/slurm/sweep_obs_augment.sh
#   TASK=cup OBS_NOISE_MAX=0.02 OBS_DROPOUT_PROB=0.15 bash scripts/slurm/sweep_obs_augment.sh
# ══════════════════════════════════════════════════════════════════

set -e

TASK=${TASK:-cup}
NOISE=${OBS_NOISE_MAX:-0.01}
DROP=${OBS_DROPOUT_PROB:-0.1}
SCRIPT=scripts/slurm/train_td3.sh

echo "=== Object State Augmentation Ablation ==="
echo "Task: $TASK | noise_max=$NOISE | dropout_prob=$DROP"
echo ""

# 1. both (noise + dropout)
echo "[1/4] Submitting: both (noise=$NOISE + dropout=$DROP)"
OBS_NOISE_MAX=$NOISE OBS_DROPOUT_PROB=$DROP TASK=$TASK \
    WANDB_NAME="${TASK}_aug_both" \
    sbatch --job-name="${TASK}_aug_both" $SCRIPT

# 2. noise only
echo "[2/4] Submitting: noise only (noise=$NOISE)"
OBS_NOISE_MAX=$NOISE TASK=$TASK \
    WANDB_NAME="${TASK}_aug_noise" \
    sbatch --job-name="${TASK}_aug_noise" $SCRIPT

# 3. dropout only
echo "[3/4] Submitting: dropout only (dropout=$DROP)"
OBS_DROPOUT_PROB=$DROP TASK=$TASK \
    WANDB_NAME="${TASK}_aug_drop" \
    sbatch --job-name="${TASK}_aug_drop" $SCRIPT

# 4. clean (baseline — no augmentation)
echo "[4/4] Submitting: clean (no augmentation)"
TASK=$TASK \
    WANDB_NAME="${TASK}_aug_clean" \
    sbatch --job-name="${TASK}_aug_clean" $SCRIPT

echo ""
echo "=== 4 jobs submitted. Monitor with: squeue -u \$(whoami) ==="
