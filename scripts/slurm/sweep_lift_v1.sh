#!/bin/bash
# Lift sweep v1: 8 configs on core-own
# Axes: reward (dense_v2/sparse), offline_frac (0.5/0.0), action_scale (0.05/0.1/0.15), l2_reg (1/5/10)
set -e
cd ~/Repos/Intern/residual-offpolicy-rl

S=scripts/slurm/train_td3.sh

# Job 1: dense_v2, offline=0.5, scale=0.1, L2=10 (baseline)
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.5 ACTION_SCALE=0.1 ACTION_L2=10.0 \
  WANDB_NAME=lift_v1_dense2_off50_s10_l10 \
  sbatch --job-name=lift_v1 $S

# Job 2: dense_v2, offline=0.5, scale=0.05, L2=10 (conservative scale)
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.5 ACTION_SCALE=0.05 ACTION_L2=10.0 \
  WANDB_NAME=lift_v1_dense2_off50_s05_l10 \
  sbatch --job-name=lift_v1 $S

# Job 3: dense_v2, offline=0.5, scale=0.15, L2=10 (aggressive scale)
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.5 ACTION_SCALE=0.15 ACTION_L2=10.0 \
  WANDB_NAME=lift_v1_dense2_off50_s15_l10 \
  sbatch --job-name=lift_v1 $S

# Job 4: dense_v2, online only, scale=0.1, L2=10
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.0 ACTION_SCALE=0.1 ACTION_L2=10.0 \
  WANDB_NAME=lift_v1_dense2_online_s10_l10 \
  sbatch --job-name=lift_v1 $S

# Job 5: dense_v2, online, scale=0.15, L2=5 (aggressive + mid L2)
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.0 ACTION_SCALE=0.15 ACTION_L2=5.0 \
  WANDB_NAME=lift_v1_dense2_online_s15_l5 \
  sbatch --job-name=lift_v1 $S

# Job 6: dense_v2, offline=0.5, scale=0.1, L2=1 (relaxed L2)
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.5 ACTION_SCALE=0.1 ACTION_L2=1.0 \
  WANDB_NAME=lift_v1_dense2_off50_s10_l1 \
  sbatch --job-name=lift_v1 $S

# Job 7: sparse, offline=0.5, scale=0.1, L2=10
TASK=lift REWARD_TYPE=sparse OFFLINE_FRAC=0.5 ACTION_SCALE=0.1 ACTION_L2=10.0 \
  WANDB_NAME=lift_v1_sparse_off50_s10_l10 \
  sbatch --job-name=lift_v1 $S

# Job 8: sparse, online, scale=0.1, L2=5
TASK=lift REWARD_TYPE=sparse OFFLINE_FRAC=0.0 ACTION_SCALE=0.1 ACTION_L2=5.0 \
  WANDB_NAME=lift_v1_sparse_online_s10_l5 \
  sbatch --job-name=lift_v1 $S

echo "Submitted 8 lift_v1 sweep jobs"
