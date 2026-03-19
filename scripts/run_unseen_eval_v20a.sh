#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"

cd /workspace/isaaclab
exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/eval_single_run.py \
  --headless --enable_cameras \
  --checkpoint "/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/pickMushroom_resid_v20a_vlm_fix_only/20260317_122008/checkpoints/best_agent.pt" \
  --env_type unseen \
  --run_label v20a_best_unseen \
  --output_dir "/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/unseen_eval_v16_vs_v20a/v20a_best_unseen" \
  --csv_base_dir "/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758" \
  --groot_model_path "/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000" \
  --groot_embodiment_tag "new_embodiment" \
  --groot_policy_device "cuda:0" \
  --language_override "pick up mushroom" \
  --max_episode_steps 1000 \
  --success_threshold 0.03
