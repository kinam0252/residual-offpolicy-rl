#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_train_residual_td3_isaaclab.sh"
CONFIG_DIR="${SCRIPT_DIR}/../configs"
DEFAULT_CONFIG_FILE="${CONFIG_DIR}/coffee_default.env"
CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "[ERROR] runner not found: ${RUNNER}" >&2
  exit 1
fi

SMOKE="${SMOKE:-0}"

if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
  set +a
fi

# Runtime defaults (override by exporting env vars before execution)
export HEADLESS="${HEADLESS:-1}"
export RUN_IN_DOCKER="${RUN_IN_DOCKER:-1}"
export DOCKER_CONTAINER="${DOCKER_CONTAINER:-isaaclab_resfit}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-$HOME/Keys/wandb_token}"
export NUM_ENVS="${NUM_ENVS:-1}"
export HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC:-15}"
export STACK_DUMP_INTERVAL_SEC="${STACK_DUMP_INTERVAL_SEC:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-draftrec}"
export WANDB_ENTITY="${WANDB_ENTITY:-kkn18}"
export WANDB_NAME="${WANDB_NAME:-resfit-isaacsim}"
export WANDB_GROUP="${WANDB_GROUP:-resfit}"
export WANDB_NOTES="${WANDB_NOTES:-}"
export WANDB_CONTINUE_RUN_ID="${WANDB_CONTINUE_RUN_ID:-}"
export WANDB_LOG_EVERY_STEPS="${WANDB_LOG_EVERY_STEPS:-10}"
export EVAL_INTERVAL_EVERY_STEPS="${EVAL_INTERVAL_EVERY_STEPS:-}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-}"
export EVAL_FIRST="${EVAL_FIRST:-0}"
export SAVE_VIDEO="${SAVE_VIDEO:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"

if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
  token_line="$(grep -m1 -v '^[[:space:]]*$' "${WANDB_API_KEY_FILE}" | tr -d '\r\n')"
  if [[ "${token_line}" == *"WANDB_API_KEY="* ]]; then
    token_line="${token_line#*WANDB_API_KEY=}"
  fi
  token_line="${token_line%\"}"
  token_line="${token_line#\"}"
  token_line="${token_line%\'}"
  token_line="${token_line#\'}"
  if [[ -n "${token_line}" ]]; then
    WANDB_API_KEY="${token_line}"
    export WANDB_API_KEY
  fi
fi

# Data / model defaults
export CSV_BASE_DIR="${CSV_BASE_DIR:-/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_101454_058}"
export GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000}"
export LANGUAGE_OVERRIDE="${LANGUAGE_OVERRIDE:-pick up mushroom}"

# Train-step defaults
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
export LEARNING_STARTS="${LEARNING_STARTS:-10000}"
export CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-10000}"
export UPDATE_EVERY_N_STEPS="${UPDATE_EVERY_N_STEPS:-4}"
export NUM_UPDATES_PER_ITERATION="${NUM_UPDATES_PER_ITERATION:-4}"

# TD3 hyperparameter defaults (coffee README-inspired)
OFFLINE_FRACTION="${OFFLINE_FRACTION:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUFFER_SIZE="${BUFFER_SIZE:-300000}"
GAMMA="${GAMMA:-0.995}"
N_STEP="${N_STEP:-5}"
STDDEV_MAX="${STDDEV_MAX:-0.025}"
STDDEV_MIN="${STDDEV_MIN:-0.025}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ACTION_SCALE="${ACTION_SCALE:-0.2}"
RANDOM_ACTION_NOISE_SCALE="${RANDOM_ACTION_NOISE_SCALE:-0.1}"

if [[ "${SMOKE}" == "1" ]]; then
  export WANDB_MODE="disabled"
  export TOTAL_TIMESTEPS="2"
  export EVAL_FIRST="1"
  export EVAL_NUM_EPISODES="1"
  export SAVE_VIDEO="0"
  export OFFLINE_FRACTION="0"
  export LEARNING_STARTS="1"
  export CRITIC_WARMUP_STEPS="1"
  export MAX_EPISODE_STEPS="1000"
  export STACK_DUMP_INTERVAL_SEC="0"
fi

if [[ ! -d "${CSV_BASE_DIR}" ]]; then
  echo "[ERROR] CSV_BASE_DIR does not exist: ${CSV_BASE_DIR}" >&2
  exit 1
fi

if [[ ! -e "${GROOT_MODEL_PATH}" ]]; then
  echo "[ERROR] GROOT_MODEL_PATH does not exist: ${GROOT_MODEL_PATH}" >&2
  exit 1
fi

exec bash "${RUNNER}" \
  --offline_fraction "${OFFLINE_FRACTION}" \
  --batch_size "${BATCH_SIZE}" \
  --buffer_size "${BUFFER_SIZE}" \
  --gamma "${GAMMA}" \
  --n_step "${N_STEP}" \
  --stddev_max "${STDDEV_MAX}" \
  --stddev_min "${STDDEV_MIN}" \
  --actor_lr "${ACTOR_LR}" \
  --action_scale "${ACTION_SCALE}" \
  --random_action_noise_scale "${RANDOM_ACTION_NOISE_SCALE}" \
  "$@"
