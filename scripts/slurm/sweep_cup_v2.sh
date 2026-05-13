#!/bin/bash
# Cup sweep v2: 8 jobs based on cross-task best configs
# Reward: dense_v3 (6 jobs) + sparse (2 jobs)
# Fixed: gamma=0.95, 500K steps, 27 envs
set -e
cd ~/Repos/Intern/residual-offpolicy-rl

S=scripts/slurm/train_td3.sh
QOS="--qos=core-extra"
RES="--cpus-per-task=14 --mem=80G --gres=gpu:1 --time=2-00:00:00"

# Job 1: conservative baseline — s=0.1, L2=1.0
TASK=cup REWARD_TYPE=dense_v3 OFFLINE_FRAC=0.75 ACTION_SCALE=0.1 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=cup_s01_l1 \
  sbatch --job-name=cup_v2 $QOS $RES $S

# Job 2: PnP best pattern — s=0.2, L2=1.0
TASK=cup REWARD_TYPE=dense_v3 OFFLINE_FRAC=0.75 ACTION_SCALE=0.2 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=cup_s02_l1 \
  sbatch --job-name=cup_v2 $QOS $RES $S

# Job 3: Lift v2 best — s=0.3, L2=1.0
TASK=cup REWARD_TYPE=dense_v3 OFFLINE_FRAC=0.75 ACTION_SCALE=0.3 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=cup_s03_l1 \
  sbatch --job-name=cup_v2 $QOS $RES $S

# Job 4: Lift v2 bold — s=0.5, L2=1.0
TASK=cup REWARD_TYPE=dense_v3 OFFLINE_FRAC=0.75 ACTION_SCALE=0.5 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=cup_s05_l1 \
  sbatch --job-name=cup_v2 $QOS $RES $S

# Job 5: L2 절반 — s=0.2, L2=0.5
TASK=cup REWARD_TYPE=dense_v3 OFFLINE_FRAC=0.75 ACTION_SCALE=0.2 ACTION_L2=0.5 GAMMA=0.95 \
  WANDB_NAME=cup_s02_l05 \
  sbatch --job-name=cup_v2 $QOS $RES $S

# Job 6: L2=0 aggressive — s=0.3, L2=0, online↑
TASK=cup REWARD_TYPE=dense_v3 OFFLINE_FRAC=0.5 ACTION_SCALE=0.3 ACTION_L2=0.0 GAMMA=0.95 \
  WANDB_NAME=cup_s03_l0 \
  sbatch --job-name=cup_v2 $QOS $RES $S

# Job 7: sparse + PnP scale — s=0.2, L2=1.0
TASK=cup REWARD_TYPE=sparse OFFLINE_FRAC=0.75 ACTION_SCALE=0.2 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=cup_s02_sparse \
  sbatch --job-name=cup_v2 $QOS $RES $S

# Job 8: sparse + bigger scale — s=0.3, L2=1.0, online↑
TASK=cup REWARD_TYPE=sparse OFFLINE_FRAC=0.5 ACTION_SCALE=0.3 ACTION_L2=1.0 GAMMA=0.95 \
  WANDB_NAME=cup_s03_sparse \
  sbatch --job-name=cup_v2 $QOS $RES $S

echo "Submitted 8 cup_v2 sweep jobs"
