#!/usr/bin/env bash
set -euo pipefail

# Workspace-local launcher for:
#   franka_cogact_motion_generation_abs_window_gr00t_training_only_basic.py

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${WORKSPACE_DIR}/franka_cogact_motion_generation_abs_window_gr00t_training_only_basic.py"
LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/logs}"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/gr00t_basic_${RUN_TS}.log"

# Isaac Lab root / launcher
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${WORKSPACE_DIR}/../../Honda_IsaacLab}"
ISAACLAB_SH="${ISAACLAB_SH:-${ISAACLAB_ROOT}/isaaclab.sh}"

# GR00T package path
ISAAC_GROOT_PATH="${ISAAC_GROOT_PATH:-${WORKSPACE_DIR}/../../Isaac-GR00T}"

# Isolated workspace deps (does not touch existing environments)
ISO_DEPS_DIR="${ISO_DEPS_DIR:-${WORKSPACE_DIR}/.pydeps_groot_iso}"
ISO_VENV_DIR="${ISO_VENV_DIR:-${WORKSPACE_DIR}/.venv_groot_iso}"
ISO_SITE_PACKAGES=""

if [[ -d "${ISO_DEPS_DIR}" ]]; then
  export PYTHONPATH="${ISO_DEPS_DIR}:${ISAAC_GROOT_PATH}:${PYTHONPATH:-}"
  echo "[INFO] Using isolated deps dir: ${ISO_DEPS_DIR}"
else
  if [[ -d "${ISO_VENV_DIR}" ]]; then
    for d in "${ISO_VENV_DIR}"/lib/python*/site-packages; do
      if [[ -d "$d" ]]; then
        ISO_SITE_PACKAGES="$d"
        break
      fi
    done
  fi

  if [[ -n "${ISO_SITE_PACKAGES}" ]]; then
    export PYTHONPATH="${ISO_SITE_PACKAGES}:${ISAAC_GROOT_PATH}:${PYTHONPATH:-}"
    echo "[INFO] Using isolated site-packages: ${ISO_SITE_PACKAGES}"
  else
    export PYTHONPATH="${ISAAC_GROOT_PATH}:${PYTHONPATH:-}"
    echo "[WARN] Isolated deps not found at ${ISO_DEPS_DIR}"
    echo "[WARN] Run: bash ${WORKSPACE_DIR}/setup_groot_isolated_env.sh"
  fi
fi

# Default episode CSV folder (can be overridden by env var CSV_DIR)
CSV_DIR="${CSV_DIR:-/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom}"

# Optional knobs (override by env vars)
MODEL_PATH="${MODEL_PATH:-/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
COMPARE_INTERVAL="${COMPARE_INTERVAL:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"

# Default to headless for stability unless explicitly overridden by CLI args
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"

{
  echo "[RUN] ts=${RUN_TS}"
  echo "[RUN] log=${RUN_LOG}"
  echo "[RUN] workspace=${WORKSPACE_DIR}"
  echo "[RUN] script=${SCRIPT_PATH}"
  echo "[RUN] csv_dir=${CSV_DIR}"
  echo "[RUN] model_path=${MODEL_PATH}"
  echo "[RUN] embodiment_tag=${EMBODIMENT_TAG}"
  echo "[RUN] policy_device=${POLICY_DEVICE}"
  echo "[RUN] iso_deps_dir=${ISO_DEPS_DIR}"
  echo "[RUN] headless_flag=${HEADLESS_FLAG}"
  echo "[RUN] extra_args=$*"
} | tee -a "${RUN_LOG}"

if command -v free >/dev/null 2>&1; then
  {
    echo "[SYS] free -h (before)"
    free -h
  } | tee -a "${RUN_LOG}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  {
    echo "[SYS] nvidia-smi (before)"
    nvidia-smi
  } | tee -a "${RUN_LOG}"
fi

if [[ ! -f "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] isaaclab.sh not found: ${ISAACLAB_SH}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] script not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

set +e
PYTHONUNBUFFERED=1 \
PYTHONFAULTHANDLER=1 \
"${ISAACLAB_SH}" -p "${SCRIPT_PATH}" \
  ${HEADLESS_FLAG} \
  --csv_dir "${CSV_DIR}" \
  --model_path "${MODEL_PATH}" \
  --embodiment_tag "${EMBODIMENT_TAG}" \
  --policy_device "${POLICY_DEVICE}" \
  --compare_debug_dump \
  --compare_debug_interval "${COMPARE_INTERVAL}" \
  --max_attempts_per_episode "${MAX_ATTEMPTS}" \
  "$@" 2>&1 | tee -a "${RUN_LOG}"
rc=${PIPESTATUS[0]}
set -e

{
  echo "[RUN] exit_code=${rc}"
  echo "[RUN] finished_ts=$(date +%Y%m%d_%H%M%S)"
} | tee -a "${RUN_LOG}"

if command -v free >/dev/null 2>&1; then
  {
    echo "[SYS] free -h (after)"
    free -h
  } | tee -a "${RUN_LOG}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  {
    echo "[SYS] nvidia-smi (after)"
    nvidia-smi
  } | tee -a "${RUN_LOG}"
fi

if [[ ${rc} -ne 0 ]]; then
  {
    echo "[DIAG] process snapshot"
    pgrep -af "franka_cogact_motion_generation_abs_window_gr00t_training_only_basic.py|isaac-sim|kit/python/bin/python3" || true
  } | tee -a "${RUN_LOG}"

  if command -v dmesg >/dev/null 2>&1; then
    {
      echo "[DIAG] dmesg OOM/kill tail"
      dmesg 2>/dev/null | grep -Ei "killed process|out of memory|oom" | tail -n 80 || true
    } | tee -a "${RUN_LOG}"
  fi

  echo "[DIAG] failure log saved: ${RUN_LOG}" | tee -a "${RUN_LOG}"
fi

exit ${rc}
