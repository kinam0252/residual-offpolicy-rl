#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Launch Residual TD3 training on IsaacLab
# Usage:  bash scripts/run_train.sh
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Load hyperparameters ──
source "${SCRIPT_DIR}/config_train.sh"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Residual TD3 — ${EXP_NAME}"
echo "║  W&B: ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_MODE})"
echo "║  Train envs: ${NUM_TRAIN_ENVS}  Eval envs: ${EVAL_NUM_ENVS}"
echo "║  Total steps: ${TOTAL_TIMESTEPS}  Seed: ${SEED}"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── Prepare output dir & log ──
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Log: ${LOG_FILE}"

# ── Kill any existing training process ──
docker exec "${CONTAINER}" pkill -9 -f train_residual_td3_isaaclab 2>/dev/null || true
sleep 2

# ── Build optional flags ──
OPT_FLAGS=""
[[ "${EVAL_FIRST}" == "true" ]] && OPT_FLAGS+=" --eval_first"
[[ "${SAVE_VIDEO}" == "true" ]] && OPT_FLAGS+=" --save_video"
[[ "${EVAL_SAVE_VIDEO}" == "true" ]] && OPT_FLAGS+=" --eval_save_video"
[[ "${DEBUG_ZERO_RESIDUAL}" == "true" ]] && OPT_FLAGS+=" --debug_zero_residual"
[[ "${DISABLE_EVAL}" == "true" ]] && OPT_FLAGS+=" --disable_eval"

# ── Write launcher script (avoids all quoting issues with docker exec) ──
LAUNCHER="${OUTPUT_DIR}/_launch.sh"
cat > "${LAUNCHER}" <<'HEREDOC_END'
#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
HEREDOC_END

cat >> "${LAUNCHER}" <<HEREDOC_VARS
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
export WANDB_API_KEY="${WANDB_API_KEY}"
cd /workspace/isaaclab
exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/train_residual_td3_isaaclab.py \\
  --headless \\
  --num_envs ${NUM_TRAIN_ENVS} \\
  --eval_num_envs ${EVAL_NUM_ENVS} \\
  --total_timesteps ${TOTAL_TIMESTEPS} \\
  --learning_starts ${LEARNING_STARTS} \\
  --critic_warmup_steps ${CRITIC_WARMUP_STEPS} \\
  --batch_size ${BATCH_SIZE} \\
  --buffer_size ${BUFFER_SIZE} \\
  --gamma ${GAMMA} \\
  --n_step ${N_STEP} \\
  --offline_fraction ${OFFLINE_FRACTION} \\
  --offline_data_dir "${OFFLINE_DATA_DIR}" \\
  --success_threshold ${SUCCESS_THRESHOLD} \\
  --cube_perturb_range ${CUBE_PERTURB_RANGE:-0} \\
  ${CUBE_PERTURB_TABLE:+--cube_perturb_table_path "${CUBE_PERTURB_TABLE}"} \\
  ${REWARD_TYPE:+--reward_type ${REWARD_TYPE}} \\
  --random_action_noise_scale ${RANDOM_ACTION_NOISE_SCALE} \\
  --stddev_max ${STDDEV_MAX} \\
  --stddev_min ${STDDEV_MIN} \\
  --actor_lr ${ACTOR_LR} \\
  --action_scale ${ACTION_SCALE} \\
  --max_episode_steps ${MAX_EPISODE_STEPS} \\
  --seed ${SEED} \\
  --csv_base_dir "${CSV_BASE_DIR}" \\
  --groot_model_path "${GROOT_MODEL_PATH}" \\
  --groot_embodiment_tag "${GROOT_EMBODIMENT_TAG}" \\
  --groot_policy_device "${GROOT_POLICY_DEVICE}" \\
  --language_override "${LANGUAGE_OVERRIDE}" \\
  --eval_interval_every_steps ${EVAL_INTERVAL} \\
  --wandb_mode "${WANDB_MODE}" \\
  --wandb_project "${WANDB_PROJECT}" \\
  --wandb_entity "${WANDB_ENTITY}" \\
  --wandb_name "${EXP_NAME}" \\
  --wandb_group "${EXP_GROUP}" \\
  --wandb_notes "${EXP_NOTES}" \\
  --wandb_log_every_steps ${WANDB_LOG_EVERY} \\
  --output_dir "${OUTPUT_DIR}" \\
  --device "${DEVICE}" \\
  ${OPT_FLAGS}
HEREDOC_VARS
chmod +x "${LAUNCHER}"

echo ""
echo "Launching in foreground (Ctrl+C to stop)..."
echo "Log: ${LOG_FILE}"
echo ""

# ── Run: foreground docker exec piped through tee ──
# Foreground = errors appear immediately, exit code propagates.
docker exec "${CONTAINER}" bash "${LAUNCHER}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "Training finished successfully."
else
  echo "Training FAILED (exit code ${EXIT_CODE}). Last 30 lines:"
  tail -30 "${LOG_FILE}"
fi
exit ${EXIT_CODE}
