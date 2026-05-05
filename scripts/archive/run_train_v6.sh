#!/usr/bin/env bash
# Launch v6 training (async eval mode — no inline eval)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Override config source
source "${SCRIPT_DIR}/config_train_v6.sh"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Residual TD3 — ${EXP_NAME} (async eval mode)"
echo "║  Train envs: ${NUM_TRAIN_ENVS}  Inline eval: DISABLED"
echo "║  Total steps: ${TOTAL_TIMESTEPS}  Seed: ${SEED}"
echo "╚═══════════════════════════════════════════════════════════╝"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Log: ${LOG_FILE}"

docker exec "${CONTAINER}" pkill -9 -f "train_residual_td3_isaaclab.*v6_async" 2>/dev/null || true
sleep 1

# Build optional flags
OPT_FLAGS=""
[[ "${EVAL_FIRST}" == "true" ]] && OPT_FLAGS+=" --eval_first"
[[ "${SAVE_VIDEO}" == "true" ]] && OPT_FLAGS+=" --save_video"
[[ "${DEBUG_ZERO_RESIDUAL}" == "true" ]] && OPT_FLAGS+=" --debug_zero_residual"
[[ "${DISABLE_EVAL}" == "true" ]] && OPT_FLAGS+=" --disable_eval"

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
echo "Launching training (foreground via docker exec, piped to log)..."
docker exec "${CONTAINER}" bash "${LAUNCHER}" 2>&1 | tee -a "${LOG_FILE}"
