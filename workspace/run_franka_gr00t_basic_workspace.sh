#!/usr/bin/env bash
set -euo pipefail

# Workspace-local launcher for:
#   franka_cogact_motion_generation_abs_window_gr00t_training_only_basic.py

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${WORKSPACE_DIR}/franka_cogact_motion_generation_abs_window_gr00t_training_only_basic.py"

# Isaac Lab root (adjust only if your repo location changes)
ISAACLAB_ROOT="/home/kinam/Desktop/Repos/VLA_RL/Honda_IsaacLab"
ISAACLAB_SH="${ISAACLAB_ROOT}/isaaclab.sh"

# GR00T package path
ISAAC_GROOT_PATH="/home/kinam/Desktop/Repos/VLA_RL/Isaac-GR00T"

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
CSV_DIR="${CSV_DIR:-/home/kinam/Desktop/DATA/dataset_from_Namiko/0_Raw_dataset/pickMushroom/pickMushroom_20251209_081744_174}"

# Optional knobs (override by env vars)
MODEL_PATH="${MODEL_PATH:-/home/kinam/Desktop/Repos/VLA_RL/Isaac-GR00T/outputs/checkpoint-100000}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
COMPARE_INTERVAL="${COMPARE_INTERVAL:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"

if [[ ! -f "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] isaaclab.sh not found: ${ISAACLAB_SH}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] script not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

"${ISAACLAB_SH}" -p "${SCRIPT_PATH}" \
  --csv_dir "${CSV_DIR}" \
  --model_path "${MODEL_PATH}" \
  --embodiment_tag "${EMBODIMENT_TAG}" \
  --policy_device "${POLICY_DEVICE}" \
  --compare_debug_dump \
  --compare_debug_interval "${COMPARE_INTERVAL}" \
  --max_attempts_per_episode "${MAX_ATTEMPTS}" \
  "$@"
