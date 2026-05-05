#!/usr/bin/env bash
# Launch 4 separate Isaac Sim instances for 4-way eval
set -uo pipefail

CONTAINER="isaaclab_resfit"
CKPT_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/pickMushroom_resid_v16_l2reg/20260315_200019/checkpoints"
CSV="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758"
GROOT="/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000"
SCRIPT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/eval_single_run.py"
OUT_BASE="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/eval_4way_v16"
PYPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"

mkdir -p "$OUT_BASE"

RUNS=(
  "agent_step0.pt|train|step0_train"
  "agent_step2500.pt|train|step2500_train"
  "agent_step0.pt|unseen|step0_unseen"
  "agent_step2500.pt|unseen|step2500_unseen"
)

for RUN in "${RUNS[@]}"; do
  IFS='|' read -r CKPT ENV_TYPE LABEL <<< "$RUN"
  LOG="$OUT_BASE/${LABEL}.log"
  LAUNCHER="$OUT_BASE/_launch_${LABEL}.sh"

  cat > "$LAUNCHER" << EOF
#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PYPATH"
cd /workspace/isaaclab
exec ./isaaclab.sh -p $SCRIPT \\
  --headless --enable_cameras \\
  --checkpoint "$CKPT_DIR/$CKPT" \\
  --env_type "$ENV_TYPE" \\
  --run_label "$LABEL" \\
  --csv_base_dir "$CSV" \\
  --groot_model_path "$GROOT" \\
  --language_override "pick up mushroom" \\
  --max_episode_steps 1000 \\
  --success_threshold 0.03 \\
  --output_dir "$OUT_BASE/$LABEL"
EOF
  chmod +x "$LAUNCHER"
  docker exec -d "$CONTAINER" bash -c "bash '$LAUNCHER' > '$LOG' 2>&1"
  echo "  Launched: $LABEL (log: $LOG)"
done

echo ""
echo "All 4 evals launched in parallel!"
echo "Monitor: for L in step0_train step2500_train step0_unseen step2500_unseen; do echo \"=== \$L ===\"; tail -3 $OUT_BASE/\${L}.log 2>/dev/null; done"
echo "Kill all: docker exec $CONTAINER pkill -9 -f eval_single_run"
