#!/bin/bash
# Lift sweep v2: Based on PnP/Stack best configs
# Key changes vs v1: L2 10→0~5, gamma 0.99→0.9~0.95, scale 0.1→0.2~0.5
set -e
cd ~/Repos/Intern/residual-offpolicy-rl

S=scripts/slurm/train_td3.sh

# Job 1: PnP best config (86.7% SR) — L2=1, γ=0.95, off=0.75, s=0.2
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.75 ACTION_SCALE=0.2 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=lift_v2_pnp_best \
  sbatch --job-name=lift_v2 $S

# Job 2: PnP 2nd + Stack style — L2=3, γ=0.95, off=0.5, s=0.2
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.5 ACTION_SCALE=0.2 ACTION_L2=3.0 GAMMA=0.95 \
  WANDB_NAME=lift_v2_l2_3_g95 \
  sbatch --job-name=lift_v2 $S

# Job 3: PnP best + larger scale — L2=1, γ=0.95, off=0.75, s=0.3
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.75 ACTION_SCALE=0.3 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=lift_v2_s03_l1 \
  sbatch --job-name=lift_v2 $S

# Job 4: Stack best config (82% SR) — L2=5, γ=0.95, off=0.5, s=0.2
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.5 ACTION_SCALE=0.2 ACTION_L2=5.0 GAMMA=0.95 \
  WANDB_NAME=lift_v2_stack_best \
  sbatch --job-name=lift_v2 $S

# Job 5: PnP γ=0.9 계열 — L2=3, γ=0.9, off=0.5, s=0.3
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.5 ACTION_SCALE=0.3 ACTION_L2=3.0 GAMMA=0.9 \
  WANDB_NAME=lift_v2_s03_l3_g9 \
  sbatch --job-name=lift_v2 $S

# Job 6: L2=0 aggressive — γ=0.95, off=0.5, s=0.2
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.5 ACTION_SCALE=0.2 ACTION_L2=0.0 GAMMA=0.95 \
  WANDB_NAME=lift_v2_no_l2 \
  sbatch --job-name=lift_v2 $S

# Job 7: Bold scale — L2=1, γ=0.95, off=0.75, s=0.5
TASK=lift REWARD_TYPE=dense_v2 OFFLINE_FRAC=0.75 ACTION_SCALE=0.5 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=lift_v2_s05_bold \
  sbatch --job-name=lift_v2 $S

echo "Submitted 7 lift_v2 sweep jobs"
