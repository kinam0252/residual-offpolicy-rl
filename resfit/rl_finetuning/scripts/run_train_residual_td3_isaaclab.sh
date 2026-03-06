#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_RL_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
RESFIT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ISAACLAB_SH="${VLA_RL_ROOT}/Honda_IsaacLab/isaaclab.sh"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_residual_td3_isaaclab.py"
ISAAC_GROOT_ROOT="${VLA_RL_ROOT}/Isaac-GR00T"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
  echo "[ERROR] isaaclab launcher not found or not executable: ${ISAACLAB_SH}" >&2
  exit 1
fi

if [[ ! -d "${ISAAC_GROOT_ROOT}" ]]; then
  echo "[ERROR] Isaac-GR00T repo not found: ${ISAAC_GROOT_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${RESFIT_REPO_ROOT}:${ISAAC_GROOT_ROOT}:${PYTHONPATH:-}"

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
)

if [[ "${HEADLESS:-0}" == "1" ]]; then
  cmd+=(--headless)
fi

if [[ -n "${CSV_BASE_DIR:-}" ]]; then
  cmd+=(--csv_base_dir "${CSV_BASE_DIR}")
fi

if [[ -n "${LANGUAGE_OVERRIDE:-}" ]]; then
  cmd+=(--language_override "${LANGUAGE_OVERRIDE}")
fi

if [[ -n "${TASK_DESCRIPTION:-}" ]]; then
  cmd+=(--task_description "${TASK_DESCRIPTION}")
fi

exec "${cmd[@]}" "$@"
