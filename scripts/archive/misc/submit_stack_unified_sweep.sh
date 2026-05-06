#!/bin/bash
# Stack unified sweep — 9 experiments across 3 partitions
# AS × L2 × γ × OF sweep, unified code with ActionScaler + replay fix
set -e
cd ~/Repos/Intern/residual-offpolicy-rl

SCRIPT=scripts/slurm_train_td3.sh

# Common overrides for all stack experiments
COMMON="TASK=stack"

# ═══════════════════════════════════════════════════════════════
# Experiment grid (9 total)
# ═══════════════════════════════════════════════════════════════
# Format: NAME AS L2 GAMMA OF
EXPS=(
  "stk_u1  0.1   0.0  0.95  0.5"
  "stk_u2  0.1   0.1  0.95  0.5"
  "stk_u3  0.1   1.0  0.95  0.75"
  "stk_u4  0.05  0.0  0.95  0.5"
  "stk_u5  0.05  0.1  0.95  0.75"
  "stk_u6  0.05  1.0  0.95  0.75"
  "stk_u7  0.1   0.0  0.99  0.5"
  "stk_u8  0.05  0.0  0.99  0.75"
  "stk_u9  0.2   0.1  0.95  0.5"
)

# Partition assignment: 1-3 core-extra, 4-6 core-on-sub, 7-9 core-on-free
PARTITIONS=(
  "core --qos=core-extra"
  "core --qos=core-extra"
  "core --qos=core-extra"
  "sub --qos=core-on-sub"
  "sub --qos=core-on-sub"
  "sub --qos=core-on-sub"
  "free --qos=core-on-free"
  "free --qos=core-on-free"
  "free --qos=core-on-free"
)

echo "═══════════════════════════════════════════"
echo "  Stack Unified Sweep — 9 experiments"
echo "═══════════════════════════════════════════"

for i in "${!EXPS[@]}"; do
  read -r NAME AS L2 GAMMA OF <<< "${EXPS[$i]}"
  read -r PART QOS <<< "${PARTITIONS[$i]}"

  echo "[$((i+1))/9] $NAME: AS=$AS L2=$L2 γ=$GAMMA OF=$OF → $PART $QOS"

  TASK=stack ACTION_SCALE=$AS ACTION_L2=$L2 GAMMA=$GAMMA OFFLINE_FRAC=$OF WANDB_NAME=$NAME \
  sbatch \
    --job-name="$NAME" \
    --partition="$PART" \
    $QOS \
    "$SCRIPT"
done

echo ""
echo "All 9 jobs submitted. Check: squeue -u \$USER | grep stk_u"
