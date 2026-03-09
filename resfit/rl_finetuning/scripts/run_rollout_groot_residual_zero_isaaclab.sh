#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_RL_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
RESFIT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ISAACLAB_SH="${VLA_RL_ROOT}/Honda_IsaacLab/isaaclab.sh"
ROLLOUT_SCRIPT="${SCRIPT_DIR}/rollout_groot_residual_zero_isaaclab.py"
ISAAC_GROOT_ROOT="${VLA_RL_ROOT}/Isaac-GR00T"
ISO_DEPS_DIR="${ISO_DEPS_DIR:-${VLA_RL_ROOT}/residual-offpolicy-rl/workspace/.pydeps_resfit_train}"
DOCKER_ISO_DEPS_DIR="${DOCKER_ISO_DEPS_DIR:-/workspace/isaaclab/workspace/.pydeps_resfit_train}"
GROOT_ISO_DEPS_DIR="${GROOT_ISO_DEPS_DIR:-${VLA_RL_ROOT}/residual-offpolicy-rl/workspace/.pydeps_groot_iso}"
DOCKER_GROOT_ISO_DEPS_DIR="${DOCKER_GROOT_ISO_DEPS_DIR:-/workspace/isaaclab/workspace/.pydeps_groot_iso}"
USE_GROOT_ISO_DEPS="${USE_GROOT_ISO_DEPS:-0}"
RUN_IN_DOCKER="${RUN_IN_DOCKER:-1}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-isaaclab_resfit}"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] isaaclab launcher not found: ${ISAACLAB_SH}" >&2
  exit 1
fi

if [[ ! -d "${ISAAC_GROOT_ROOT}" ]]; then
  echo "[ERROR] Isaac-GR00T repo not found: ${ISAAC_GROOT_ROOT}" >&2
  exit 1
fi

py_parts=()
if [[ -d "${ISO_DEPS_DIR}" ]]; then
  py_parts+=("${ISO_DEPS_DIR}")
  echo "[launch] iso_deps_dir=${ISO_DEPS_DIR}"
else
  echo "[launch] iso_deps_dir=<none>"
fi
if [[ "${USE_GROOT_ISO_DEPS}" == "1" && -d "${GROOT_ISO_DEPS_DIR}" ]]; then
  py_parts+=("${GROOT_ISO_DEPS_DIR}")
  echo "[launch] groot_iso_deps_dir=${GROOT_ISO_DEPS_DIR}"
else
  echo "[launch] groot_iso_deps_dir=<disabled>"
fi
py_parts+=("${RESFIT_REPO_ROOT}" "${ISAAC_GROOT_ROOT}")
if [[ -n "${PYTHONPATH:-}" ]]; then
  py_parts+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${py_parts[*]}")"
export PYTHONUNBUFFERED=1

echo "[launch] run_rollout_groot_residual_zero_isaaclab.sh"
echo "[launch] cwd=$(pwd)"
echo "[launch] run_in_docker=${RUN_IN_DOCKER} container=${DOCKER_CONTAINER}"
echo "[launch] headless=${HEADLESS:-0} num_envs=${NUM_ENVS:-1} max_steps=${MAX_STEPS:-1000}"
echo "[launch] csv_base_dir=${CSV_BASE_DIR:-<none>}"
echo "[launch] groot_model_path=${GROOT_MODEL_PATH:-<server-mode>}"
echo "[launch] groot_policy_device=${GROOT_POLICY_DEVICE:-<default>}"
echo "[launch] docker_iso_deps_dir=${DOCKER_ISO_DEPS_DIR}"
echo "[launch] docker_groot_iso_deps_dir=${DOCKER_GROOT_ISO_DEPS_DIR}"

cmd=(
  "${ISAACLAB_SH}" -p "${ROLLOUT_SCRIPT}"
  --num_envs "${NUM_ENVS:-1}"
  --max_steps "${MAX_STEPS:-1000}"
  --gr00t_host "${GR00T_HOST:-127.0.0.1}"
  --gr00t_port "${GR00T_PORT:-5555}"
)

if [[ "${HEADLESS:-0}" == "1" ]]; then
  cmd+=(--headless)
fi

if [[ -n "${CSV_BASE_DIR:-}" ]]; then
  cmd+=(--csv_base_dir "${CSV_BASE_DIR}")
fi

if [[ -n "${CSV_INIT_ROW_INDEX:-}" ]]; then
  cmd+=(--csv_init_row_index "${CSV_INIT_ROW_INDEX}")
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

if [[ "${GROOT_POLICY_STRICT:-0}" == "1" ]]; then
  cmd+=(--groot_policy_strict)
fi

if [[ -n "${LANGUAGE_OVERRIDE:-}" ]]; then
  cmd+=(--language_override "${LANGUAGE_OVERRIDE}")
fi

if [[ -n "${TASK_DESCRIPTION:-}" ]]; then
  cmd+=(--task_description "${TASK_DESCRIPTION}")
fi

echo "[launch] exec: ${cmd[*]} $*"

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
    ./isaaclab.sh -p "${ROLLOUT_SCRIPT}"
    --num_envs "${NUM_ENVS:-1}"
    --max_steps "${MAX_STEPS:-1000}"
    --gr00t_host "${GR00T_HOST:-127.0.0.1}"
    --gr00t_port "${GR00T_PORT:-5555}"
  )

  if [[ "${HEADLESS:-0}" == "1" ]]; then
    args+=(--headless)
  fi
  if [[ -n "${CSV_BASE_DIR:-}" ]]; then
    args+=(--csv_base_dir "${CSV_BASE_DIR}")
  fi
  if [[ -n "${CSV_INIT_ROW_INDEX:-}" ]]; then
    args+=(--csv_init_row_index "${CSV_INIT_ROW_INDEX}")
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
  if [[ "${GROOT_POLICY_STRICT:-0}" == "1" ]]; then
    args+=(--groot_policy_strict)
  fi
  if [[ -n "${LANGUAGE_OVERRIDE:-}" ]]; then
    args+=(--language_override "${LANGUAGE_OVERRIDE}")
  fi
  if [[ -n "${TASK_DESCRIPTION:-}" ]]; then
    args+=(--task_description "${TASK_DESCRIPTION}")
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

  exec docker exec "${DOCKER_CONTAINER}" bash -lc "export TERM=xterm; export PYTHONUNBUFFERED=1; export PYTHONPATH='${container_pythonpath}'; cd /workspace/isaaclab; ${args_quoted}"
fi

exec "${cmd[@]}" "$@"
