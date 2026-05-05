#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab
exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/eval_single_run.py --headless --enable_cameras --checkpoint "/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/pickMushroom_resid_v16_l2reg/20260315_200019/checkpoints/agent_step47500.pt" --env_type "unseen" --run_label "step47500_unseen" --csv_base_dir "/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758" --groot_model_path "/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000" --language_override "pick up mushroom" --max_episode_steps 1000 --success_threshold 0.03 --output_dir "/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/eval_4way_v16/step47500_unseen"
