#!/usr/bin/env bash
set -e
cd /workspace/isaaclab

CKPT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/pickMushroom_resid_v10_10env/20260314_175618/checkpoints/agent_step0.pt"
CSV="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758"
GROOT="/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000"
OUTPUT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/debug_contact_viz"

exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/eval_debug_cube_perturb.py \
  --headless --enable_cameras \
  --checkpoint "$CKPT" \
  --csv_base_dir "$CSV" \
  --groot_model_path "$GROOT" \
  --language_override "pick up mushroom" \
  --max_episode_steps 1000 \
  --success_threshold 0.03 \
  --output_dir "$OUTPUT"
