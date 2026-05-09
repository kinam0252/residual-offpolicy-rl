#!/bin/bash
# Cup sweep v3: increased residual scales + dense_v3 reward
# Key changes from v2:
#   - residual_pos_scale: 0.02 → 0.05/0.10 (max 5-10mm/step vs 2mm)
#   - residual_grip_scale: 0.1 → 0.2/0.3 (grip correction ↑)
#   - dense_v3 reward: 0.25*approach + 0.15*grasp + 0.25*upright + 0.35*success
set -e
cd ~/Repos/Intern/residual-offpolicy-rl

SCRIPT=scripts/slurm/train_td3.sh

# Job 1: pos=0.05, grip=0.2, act=0.1, dense_v3, online
TASK=cup REWARD_TYPE=dense_v3 ACTION_SCALE=0.1 OFFLINE_FRAC=0.0 \
  RESIDUAL_POS_SCALE=0.05 RESIDUAL_GRIP_SCALE=0.2 \
  sbatch --job-name=cup_v3 $SCRIPT

# Job 2: pos=0.05, grip=0.2, act=0.1, dense_v3, frac=0.3
TASK=cup REWARD_TYPE=dense_v3 ACTION_SCALE=0.1 OFFLINE_FRAC=0.3 \
  RESIDUAL_POS_SCALE=0.05 RESIDUAL_GRIP_SCALE=0.2 \
  sbatch --job-name=cup_v3 $SCRIPT

# Job 3: pos=0.05, grip=0.2, act=0.1, sparse, online
TASK=cup REWARD_TYPE=sparse ACTION_SCALE=0.1 OFFLINE_FRAC=0.0 \
  RESIDUAL_POS_SCALE=0.05 RESIDUAL_GRIP_SCALE=0.2 \
  sbatch --job-name=cup_v3 $SCRIPT

# Job 4: pos=0.05, grip=0.2, act=0.15, dense_v3, online
TASK=cup REWARD_TYPE=dense_v3 ACTION_SCALE=0.15 OFFLINE_FRAC=0.0 \
  RESIDUAL_POS_SCALE=0.05 RESIDUAL_GRIP_SCALE=0.2 \
  sbatch --job-name=cup_v3 $SCRIPT

# Job 5: pos=0.10, grip=0.2, act=0.1, dense_v3, online
TASK=cup REWARD_TYPE=dense_v3 ACTION_SCALE=0.1 OFFLINE_FRAC=0.0 \
  RESIDUAL_POS_SCALE=0.10 RESIDUAL_GRIP_SCALE=0.2 \
  sbatch --job-name=cup_v3 $SCRIPT

# Job 6: pos=0.10, grip=0.2, act=0.1, sparse, online
TASK=cup REWARD_TYPE=sparse ACTION_SCALE=0.1 OFFLINE_FRAC=0.0 \
  RESIDUAL_POS_SCALE=0.10 RESIDUAL_GRIP_SCALE=0.2 \
  sbatch --job-name=cup_v3 $SCRIPT

# Job 7: pos=0.05, grip=0.3, act=0.1, dense_v3, online
TASK=cup REWARD_TYPE=dense_v3 ACTION_SCALE=0.1 OFFLINE_FRAC=0.0 \
  RESIDUAL_POS_SCALE=0.05 RESIDUAL_GRIP_SCALE=0.3 \
  sbatch --job-name=cup_v3 $SCRIPT

# Job 8: pos=0.10, grip=0.3, act=0.15, dense_v3, online
TASK=cup REWARD_TYPE=dense_v3 ACTION_SCALE=0.15 OFFLINE_FRAC=0.0 \
  RESIDUAL_POS_SCALE=0.10 RESIDUAL_GRIP_SCALE=0.3 \
  sbatch --job-name=cup_v3 $SCRIPT

echo "Submitted 8 cup_v3 sweep jobs"
