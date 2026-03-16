#!/usr/bin/env bash
# Run online critic debug for step 0 and step 2500 checkpoints
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab

CKPT_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/pickMushroom_resid_v13_offline/20260315_062235/checkpoints"
CSV="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758"
GROOT="/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000"
OUTPUT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/debug_critic_online"
SCRIPT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/debug_critic_online.py"

CKPT="${CKPT_DIR}/${1:-agent_step0.pt}"

echo "═══════════════════════════════════════════════"
echo "  Online Critic Debug: $(basename $CKPT)"
echo "═══════════════════════════════════════════════"

exec ./isaaclab.sh -p "$SCRIPT" \
  --headless --enable_cameras \
  --checkpoint "$CKPT" \
  --csv_base_dir "$CSV" \
  --groot_model_path "$GROOT" \
  --language_override "pick up mushroom" \
  --max_episode_steps 1000 \
  --success_threshold 0.03 \
  --output_dir "$OUTPUT"
