#!/usr/bin/env bash
# Generate offline data with dense_clipped reward + contact force gate
# 10 workers × 174 episodes → ~17-18 eps per worker
# Output: /home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_dense_clipped_3cm
set -uo pipefail

export N_WORKERS=10
export OUTPUT_DIR="/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_dense_clipped_3cm"
export REWARD_TYPE="dense_clipped"
export SUCCESS_THRESH="0.03"
export FRICTION="50000"
# Do NOT set CLEAN=1 to preserve existing data

echo "================================================================"
echo "  Generating offline data: dense_clipped + contact force"
echo "  Workers: ${N_WORKERS}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Reward:  ${REWARD_TYPE}"
echo "  Thresh:  ${SUCCESS_THRESH}m (3cm)"
echo "================================================================"

exec bash /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/run_replay_all.sh
