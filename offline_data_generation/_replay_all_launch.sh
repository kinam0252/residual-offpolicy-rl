#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab
exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/replay_all_csv_with_reward.py \
  --headless --enable_cameras \
  --num_envs 1 \
  --csv_base_dir "/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom" \
  --output_dir "/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_w_reward"
