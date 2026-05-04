#!/bin/bash
# Cup dense_equal_bonus sweep (submitted 2026-05-04)
# All: reward=dense_equal_bonus, offline_frac=0.5, chunk_sync, core-own
# 6 experiments: AS={0.1, 0.05} x {L2=1.0, L2=0.0, gamma=0.99}

cd ~/Repos/Intern/residual-offpolicy-rl

# 1. eq_as10_base: AS=0.1, L2=1.0, gamma=0.95
REWARD_TYPE=dense_equal_bonus ACTION_SCALE=0.1 ACTION_L2=1.0 GAMMA=0.95 OFFLINE_FRAC=0.5 WANDB_NAME=cup_eq_as10_base \
sbatch --job-name=cup_eq1 scripts/slurm_train_cup_td3.sh --chunk_sync

# 2. eq_as10_noL2: AS=0.1, L2=0.0, gamma=0.95
REWARD_TYPE=dense_equal_bonus ACTION_SCALE=0.1 ACTION_L2=0.0 GAMMA=0.95 OFFLINE_FRAC=0.5 WANDB_NAME=cup_eq_as10_noL2 \
sbatch --job-name=cup_eq2 scripts/slurm_train_cup_td3.sh --chunk_sync

# 3. eq_as10_g99: AS=0.1, L2=1.0, gamma=0.99
REWARD_TYPE=dense_equal_bonus ACTION_SCALE=0.1 ACTION_L2=1.0 GAMMA=0.99 OFFLINE_FRAC=0.5 WANDB_NAME=cup_eq_as10_g99 \
sbatch --job-name=cup_eq3 scripts/slurm_train_cup_td3.sh --chunk_sync

# 4. eq_as05_base: AS=0.05, L2=1.0, gamma=0.95
REWARD_TYPE=dense_equal_bonus ACTION_SCALE=0.05 ACTION_L2=1.0 GAMMA=0.95 OFFLINE_FRAC=0.5 WANDB_NAME=cup_eq_as05_base \
sbatch --job-name=cup_eq4 scripts/slurm_train_cup_td3.sh --chunk_sync

# 5. eq_as05_noL2: AS=0.05, L2=0.0, gamma=0.95
REWARD_TYPE=dense_equal_bonus ACTION_SCALE=0.05 ACTION_L2=0.0 GAMMA=0.95 OFFLINE_FRAC=0.5 WANDB_NAME=cup_eq_as05_noL2 \
sbatch --job-name=cup_eq5 scripts/slurm_train_cup_td3.sh --chunk_sync

# 6. eq_as05_g99: AS=0.05, L2=1.0, gamma=0.99
REWARD_TYPE=dense_equal_bonus ACTION_SCALE=0.05 ACTION_L2=1.0 GAMMA=0.99 OFFLINE_FRAC=0.5 WANDB_NAME=cup_eq_as05_g99 \
sbatch --job-name=cup_eq6 scripts/slurm_train_cup_td3.sh --chunk_sync

# Job IDs (submitted 2026-05-04 17:56):
# 6302 cup_eq1 worker-2 GPU6
# 6303 cup_eq2 worker-4 GPU0
# 6304 cup_eq3 worker-4 GPU1
# 6305 cup_eq4 (pending)
# 6306 cup_eq5 (pending)
# 6307 cup_eq6 (pending)
