#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab
exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/replay_csv_with_reward.py \
  --headless --enable_cameras \
  --num_envs 1 \
  --csv_dir "/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081744_174" \
  --output_dir "/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/offline_replay_output"
