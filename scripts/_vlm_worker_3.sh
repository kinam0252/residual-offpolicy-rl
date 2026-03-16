#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab
exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/replay_all_worker_with_groot.py \
  --headless --enable_cameras \
  --num_envs 1 \
  --csv_base_dir "/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758" \
  --output_dir "/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_dense_clipped_3cm_vlm" \
  --groot_model_path "/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000" \
  --language_override "pick up mushroom" \
  --worker_id 3 \
  --num_workers 4 \
  --friction 50000 \
  --success_threshold 0.03 \
  --reward_type dense_clipped \
  --vlm_inference_interval 16
