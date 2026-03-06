#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_RL_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
RESFIT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ISAACLAB_SH="${VLA_RL_ROOT}/Honda_IsaacLab/isaaclab.sh"
ROLLOUT_SCRIPT="${SCRIPT_DIR}/rollout_groot_residual_zero_isaaclab.py"
ISAAC_GROOT_ROOT="${VLA_RL_ROOT}/Isaac-GR00T"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] isaaclab launcher not found: ${ISAACLAB_SH}" >&2
  exit 1
fi

if [[ ! -d "${ISAAC_GROOT_ROOT}" ]]; then
  echo "[ERROR] Isaac-GR00T repo not found: ${ISAAC_GROOT_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${RESFIT_REPO_ROOT}:${ISAAC_GROOT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "[launch] run_rollout_groot_residual_zero_isaaclab.sh"
echo "[launch] cwd=$(pwd)"
echo "[launch] headless=${HEADLESS:-0} num_envs=${NUM_ENVS:-1} max_steps=${MAX_STEPS:-1200}"
echo "[launch] csv_base_dir=${CSV_BASE_DIR:-<none>}"
echo "[launch] groot_model_path=${GROOT_MODEL_PATH:-<server-mode>}"
echo "[launch] groot_policy_device=${GROOT_POLICY_DEVICE:-<default>}"

cmd=(
  "${ISAACLAB_SH}" -p "${ROLLOUT_SCRIPT}"
  --num_envs "${NUM_ENVS:-1}"
  --max_steps "${MAX_STEPS:-1200}"
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

exec "${cmd[@]}" "$@"
