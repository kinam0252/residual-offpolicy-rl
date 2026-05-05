#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Launch ASYNC eval process (separate Isaac Sim instance).
# Polls checkpoint dir for new .pt files, evaluates, logs to wandb.
# Usage:  bash scripts/run_eval_async.sh
# ═══════════════════════════════════════════════════════════════════
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Load training config (reuse same settings) ──
source "${SCRIPT_DIR}/config_train.sh"

EVAL_SCRIPT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/eval_async_isaaclab.py"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/async_eval_outputs"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Async Eval Process — ${EXP_NAME}"
echo "║  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "║  Eval envs: ${EVAL_NUM_ENVS}"
echo "╚═══════════════════════════════════════════════════════════╝"

mkdir -p "${EVAL_OUTPUT_DIR}"
LOG_FILE="${EVAL_OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

# ── Write launcher ──
LAUNCHER="${EVAL_OUTPUT_DIR}/_launch_eval.sh"
cat > "${LAUNCHER}" <<EOF
#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
export WANDB_API_KEY="${WANDB_API_KEY}"
cd /workspace/isaaclab
exec ./isaaclab.sh -p ${EVAL_SCRIPT} \\
  --headless --enable_cameras \\
  --checkpoint_dir "${CHECKPOINT_DIR}" \\
  --num_envs ${EVAL_NUM_ENVS} \\
  --max_episode_steps ${MAX_EPISODE_STEPS} \\
  --success_threshold ${SUCCESS_THRESHOLD} \\
  --groot_model_path "${GROOT_MODEL_PATH}" \\
  --groot_embodiment_tag "${GROOT_EMBODIMENT_TAG}" \\
  --groot_policy_device "${DEVICE}" \\
  --csv_base_dir "${CSV_BASE_DIR}" \\
  --language_override "${LANGUAGE_OVERRIDE}" \\
  --output_dir "${EVAL_OUTPUT_DIR}" \\
  --wandb_mode "${WANDB_MODE}" \\
  --wandb_project "${WANDB_PROJECT}" \\
  --wandb_entity "${WANDB_ENTITY}" \\
  --wandb_name "${EXP_NAME}_eval" \\
  --poll_interval_sec 30 \\
  --save_video
EOF
chmod +x "${LAUNCHER}"

echo "Launching async eval..."
echo "Log: ${LOG_FILE}"
docker exec -d "${CONTAINER}" bash "${LAUNCHER}" > "${LOG_FILE}" 2>&1 &
EVAL_PID=$!
echo "Host PID: ${EVAL_PID}"
echo "${EVAL_PID}" > /tmp/async_eval_pid.txt
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
echo "Kill:    kill ${EVAL_PID}"
