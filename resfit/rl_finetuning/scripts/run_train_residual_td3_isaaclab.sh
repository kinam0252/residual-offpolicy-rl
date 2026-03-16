#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_RL_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
RESFIT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../configs"
DEFAULT_CONFIG_FILE="${CONFIG_DIR}/coffee_default.env"
CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"

if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
  set +a
fi

LOG_DIR="${LOG_DIR:-${RESFIT_REPO_ROOT}/workspace/logs}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/resfit_train_${RUN_TS}.log}"

ISAACLAB_SH="${VLA_RL_ROOT}/Honda_IsaacLab/isaaclab.sh"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_residual_td3_isaaclab.py"
ISAAC_GROOT_ROOT="${VLA_RL_ROOT}/Isaac-GR00T"
ISO_DEPS_DIR="${ISO_DEPS_DIR:-${VLA_RL_ROOT}/residual-offpolicy-rl/workspace/.pydeps_resfit_train}"
DOCKER_ISO_DEPS_DIR="${DOCKER_ISO_DEPS_DIR:-/workspace/isaaclab/workspace/.pydeps_resfit_train}"
GROOT_ISO_DEPS_DIR="${GROOT_ISO_DEPS_DIR:-${VLA_RL_ROOT}/residual-offpolicy-rl/workspace/.pydeps_groot_iso}"
DOCKER_GROOT_ISO_DEPS_DIR="${DOCKER_GROOT_ISO_DEPS_DIR:-/workspace/isaaclab/workspace/.pydeps_groot_iso}"
USE_GROOT_ISO_DEPS="${USE_GROOT_ISO_DEPS:-0}"
RUN_IN_DOCKER="${RUN_IN_DOCKER:-1}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-isaaclab_resfit}"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] isaaclab launcher not found or not executable: ${ISAACLAB_SH}" >&2
  exit 1
fi

if [[ ! -d "${ISAAC_GROOT_ROOT}" ]]; then
  echo "[ERROR] Isaac-GR00T repo not found: ${ISAAC_GROOT_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

py_parts=()
if [[ -d "${ISO_DEPS_DIR}" ]]; then
  py_parts+=("${ISO_DEPS_DIR}")
fi
if [[ "${USE_GROOT_ISO_DEPS}" == "1" && -d "${GROOT_ISO_DEPS_DIR}" ]]; then
  py_parts+=("${GROOT_ISO_DEPS_DIR}")
fi
py_parts+=("${RESFIT_REPO_ROOT}" "${ISAAC_GROOT_ROOT}")
if [[ -n "${PYTHONPATH:-}" ]]; then
  py_parts+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${py_parts[*]}")"
export PYTHONUNBUFFERED=1

if [[ "${ALLOW_CONCURRENT_TRAIN:-0}" != "1" ]]; then
  existing_pids="$(ps -eo pid,cmd | grep -F "train_residual_td3_isaaclab.py" | grep -v grep | awk '{print $1}' | tr '\n' ' ' | xargs || true)"
  if [[ -n "${existing_pids}" ]]; then
    echo "[ERROR] Existing train_residual_td3_isaaclab.py process(es) detected: ${existing_pids}" >&2
    echo "[ERROR] Stop previous run first or set ALLOW_CONCURRENT_TRAIN=1 to override." >&2
    exit 1
  fi
fi

cmd=(
  "${ISAACLAB_SH}" -p "${TRAIN_SCRIPT}"
  --num_envs "${NUM_ENVS:-1}"
  --total_timesteps "${TOTAL_TIMESTEPS:-5000}"
  --learning_starts "${LEARNING_STARTS:-8}"
  --critic_warmup_steps "${CRITIC_WARMUP_STEPS:-8}"
  --update_every_n_steps "${UPDATE_EVERY_N_STEPS:-4}"
  --num_updates_per_iteration "${NUM_UPDATES_PER_ITERATION:-1}"
  --wandb_mode "${WANDB_MODE:-disabled}"
  --gr00t_host "${GR00T_HOST:-127.0.0.1}"
  --gr00t_port "${GR00T_PORT:-5555}"
  --heartbeat_interval_sec "${HEARTBEAT_INTERVAL_SEC:-20}"
)

if [[ "${HEADLESS:-0}" == "1" ]]; then
  cmd+=(--headless)
fi

if [[ -n "${CSV_BASE_DIR:-}" ]]; then
  cmd+=(--csv_base_dir "${CSV_BASE_DIR}")
fi
if [[ -n "${MAX_EPISODE_STEPS:-}" ]]; then
  cmd+=(--max_episode_steps "${MAX_EPISODE_STEPS}")
fi
if [[ -n "${WARMUP_COLLECTOR:-}" ]]; then
  cmd+=(--warmup_collector "${WARMUP_COLLECTOR}")
fi
if [[ -n "${STACK_DUMP_INTERVAL_SEC:-}" ]]; then
  cmd+=(--stack_dump_interval_sec "${STACK_DUMP_INTERVAL_SEC}")
fi

if [[ -n "${LANGUAGE_OVERRIDE:-}" ]]; then
  cmd+=(--language_override "${LANGUAGE_OVERRIDE}")
fi

if [[ -n "${TASK_DESCRIPTION:-}" ]]; then
  cmd+=(--task_description "${TASK_DESCRIPTION}")
fi

if [[ -n "${GROOT_MODEL_PATH:-}" ]]; then
  cmd+=(--groot_model_path "${GROOT_MODEL_PATH}")
fi

if [[ -n "${GROOT_EMBODIMENT_TAG:-}" ]]; then
  cmd+=(--groot_embodiment_tag "${GROOT_EMBODIMENT_TAG}")
fi

if [[ -n "${GROOT_POLICY_DEVICE:-}" ]]; then
  cmd+=(--groot_policy_device "${GROOT_POLICY_DEVICE}")
fi

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  cmd+=(--wandb_project "${WANDB_PROJECT}")
fi

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  cmd+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ -n "${WANDB_NAME:-}" ]]; then
  cmd+=(--wandb_name "${WANDB_NAME}")
fi

if [[ -n "${WANDB_GROUP:-}" ]]; then
  cmd+=(--wandb_group "${WANDB_GROUP}")
fi

if [[ -n "${WANDB_NOTES:-}" ]]; then
  cmd+=(--wandb_notes "${WANDB_NOTES}")
fi

if [[ -n "${WANDB_CONTINUE_RUN_ID:-}" ]]; then
  cmd+=(--wandb_continue_run_id "${WANDB_CONTINUE_RUN_ID}")
fi
  if [[ -n "${WANDB_LOG_EVERY_STEPS:-}" ]]; then
    cmd+=(--wandb_log_every_steps "${WANDB_LOG_EVERY_STEPS}")
  fi
if [[ -n "${EVAL_INTERVAL_EVERY_STEPS:-}" ]]; then
  cmd+=(--eval_interval_every_steps "${EVAL_INTERVAL_EVERY_STEPS}")
fi
if [[ -n "${EVAL_NUM_EPISODES:-}" ]]; then
  cmd+=(--eval_num_episodes "${EVAL_NUM_EPISODES}")
fi
if [[ "${EVAL_FIRST:-0}" == "1" ]]; then
  cmd+=(--eval_first)
fi
if [[ "${SAVE_VIDEO:-0}" == "1" ]]; then
  cmd+=(--save_video)
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  cmd+=(--output_dir "${OUTPUT_DIR}")
fi

if [[ "${GROOT_POLICY_STRICT:-0}" == "1" ]]; then
  cmd+=(--groot_policy_strict)
fi

if [[ "${RUN_IN_DOCKER}" == "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] RUN_IN_DOCKER=1 but docker command is unavailable." >&2
    exit 1
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${DOCKER_CONTAINER}"; then
    echo "[ERROR] Docker container not running: ${DOCKER_CONTAINER}" >&2
    exit 1
  fi

  args=(
    ./isaaclab.sh -p "${TRAIN_SCRIPT}"
    --num_envs "${NUM_ENVS:-1}"
    --total_timesteps "${TOTAL_TIMESTEPS:-5000}"
    --learning_starts "${LEARNING_STARTS:-8}"
    --critic_warmup_steps "${CRITIC_WARMUP_STEPS:-8}"
    --update_every_n_steps "${UPDATE_EVERY_N_STEPS:-4}"
    --num_updates_per_iteration "${NUM_UPDATES_PER_ITERATION:-1}"
    --wandb_mode "${WANDB_MODE:-disabled}"
    --gr00t_host "${GR00T_HOST:-127.0.0.1}"
    --gr00t_port "${GR00T_PORT:-5555}"
    --heartbeat_interval_sec "${HEARTBEAT_INTERVAL_SEC:-20}"
  )

  if [[ "${HEADLESS:-0}" == "1" ]]; then
    args+=(--headless)
  fi
  if [[ -n "${CSV_BASE_DIR:-}" ]]; then
    args+=(--csv_base_dir "${CSV_BASE_DIR}")
  fi
  if [[ -n "${MAX_EPISODE_STEPS:-}" ]]; then
    args+=(--max_episode_steps "${MAX_EPISODE_STEPS}")
  fi
  if [[ -n "${WARMUP_COLLECTOR:-}" ]]; then
    args+=(--warmup_collector "${WARMUP_COLLECTOR}")
  fi
  if [[ -n "${STACK_DUMP_INTERVAL_SEC:-}" ]]; then
    args+=(--stack_dump_interval_sec "${STACK_DUMP_INTERVAL_SEC}")
  fi
  if [[ -n "${LANGUAGE_OVERRIDE:-}" ]]; then
    args+=(--language_override "${LANGUAGE_OVERRIDE}")
  fi
  if [[ -n "${TASK_DESCRIPTION:-}" ]]; then
    args+=(--task_description "${TASK_DESCRIPTION}")
  fi
  if [[ -n "${GROOT_MODEL_PATH:-}" ]]; then
    args+=(--groot_model_path "${GROOT_MODEL_PATH}")
  fi
  if [[ -n "${GROOT_EMBODIMENT_TAG:-}" ]]; then
    args+=(--groot_embodiment_tag "${GROOT_EMBODIMENT_TAG}")
  fi
  if [[ -n "${GROOT_POLICY_DEVICE:-}" ]]; then
    args+=(--groot_policy_device "${GROOT_POLICY_DEVICE}")
  fi
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    args+=(--wandb_project "${WANDB_PROJECT}")
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    args+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_NAME:-}" ]]; then
    args+=(--wandb_name "${WANDB_NAME}")
  fi
  if [[ -n "${WANDB_GROUP:-}" ]]; then
    args+=(--wandb_group "${WANDB_GROUP}")
  fi
  if [[ -n "${WANDB_NOTES:-}" ]]; then
    args+=(--wandb_notes "${WANDB_NOTES}")
  fi
  if [[ -n "${WANDB_CONTINUE_RUN_ID:-}" ]]; then
    args+=(--wandb_continue_run_id "${WANDB_CONTINUE_RUN_ID}")
  fi
  if [[ -n "${WANDB_LOG_EVERY_STEPS:-}" ]]; then
    args+=(--wandb_log_every_steps "${WANDB_LOG_EVERY_STEPS}")
  fi
  if [[ -n "${EVAL_INTERVAL_EVERY_STEPS:-}" ]]; then
    args+=(--eval_interval_every_steps "${EVAL_INTERVAL_EVERY_STEPS}")
  fi
  if [[ -n "${EVAL_NUM_EPISODES:-}" ]]; then
    args+=(--eval_num_episodes "${EVAL_NUM_EPISODES}")
  fi
  if [[ "${EVAL_FIRST:-0}" == "1" ]]; then
    args+=(--eval_first)
  fi
  if [[ "${SAVE_VIDEO:-0}" == "1" ]]; then
    args+=(--save_video)
  fi
  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    args+=(--output_dir "${OUTPUT_DIR}")
  fi
  if [[ "${GROOT_POLICY_STRICT:-0}" == "1" ]]; then
    args+=(--groot_policy_strict)
  fi

  args+=("$@")
  args_quoted="$(printf '%q ' "${args[@]}")"

  container_pythonpath="${PYTHONPATH}"
  if [[ -n "${DOCKER_ISO_DEPS_DIR}" ]]; then
    container_pythonpath="${DOCKER_ISO_DEPS_DIR}:${container_pythonpath}"
  fi
  if [[ "${USE_GROOT_ISO_DEPS}" == "1" && -n "${DOCKER_GROOT_ISO_DEPS_DIR}" ]]; then
    container_pythonpath="${DOCKER_GROOT_ISO_DEPS_DIR}:${container_pythonpath}"
  fi

  wandb_env_exports=""
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb_env_exports+="export WANDB_API_KEY='${WANDB_API_KEY}'; "
  fi
  if [[ -n "${WANDB_ANONYMOUS:-}" ]]; then
    wandb_env_exports+="export WANDB_ANONYMOUS='${WANDB_ANONYMOUS}'; "
  fi

  exec docker exec "${DOCKER_CONTAINER}" bash -lc "export TERM=xterm; export PYTHONUNBUFFERED=1; export PYTHONPATH='${container_pythonpath}'; ${wandb_env_exports}cd /workspace/isaaclab; ${args_quoted}"
fi

exec "${cmd[@]}" "$@"
