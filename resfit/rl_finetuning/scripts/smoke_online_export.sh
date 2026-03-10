#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl"
SCRIPT_PATH="${ROOT_DIR}/resfit/rl_finetuning/scripts/export_online_episode_isaaclab.py"
LOG_PATH="${LOG_PATH:-/tmp/smoke_online_export.log}"
OUT_NPZ="${OUT_NPZ:-${ROOT_DIR}/workspace/online_episode_smoke.npz}"

DOCKER_NAME="${DOCKER_NAME:-isaaclab_resfit}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
PARSE_STALL_SEC="${PARSE_STALL_SEC:-90}"
MAX_STEPS="${MAX_STEPS:-80}"
CSV_BASE_DIR="${CSV_BASE_DIR:-/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_101454_058}"
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000}"

mkdir -p "$(dirname "${LOG_PATH}")"
rm -f "${LOG_PATH}" "${OUT_NPZ}"

echo "[SMOKE] start online export"
echo "[SMOKE] docker=${DOCKER_NAME} timeout_sec=${TIMEOUT_SEC} max_steps=${MAX_STEPS}"
echo "[SMOKE] parse_stall_sec=${PARSE_STALL_SEC}"
echo "[SMOKE] log=${LOG_PATH}"
echo "[SMOKE] out_npz=${OUT_NPZ}"

set +e
docker exec "${DOCKER_NAME}" bash -lc '
set -euo pipefail
export TERM=xterm
export PYTHONUNBUFFERED=1
export RESFIT_FAULT_DUMP_AFTER_SEC=120
export PYTHONPATH=/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/workspace/Isaac-GR00T:${PYTHONPATH:-}
cd /workspace/isaaclab
./isaaclab.sh -p '"${SCRIPT_PATH}"' \
  --headless \
  --num_envs 1 \
  --max_steps '"${MAX_STEPS}"' \
  --csv_base_dir '"${CSV_BASE_DIR}"' \
  --groot_model_path '"${GROOT_MODEL_PATH}"' \
  --groot_policy_device cuda \
  --output_npz '"${OUT_NPZ}"'
' > "${LOG_PATH}" 2>&1 &
cmd_pid=$!

start_ts=$(date +%s)
usr1_sent=0
cmd_code=0

while kill -0 "${cmd_pid}" >/dev/null 2>&1; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))

  if [[ ${elapsed} -ge ${TIMEOUT_SEC} ]]; then
    echo "[SMOKE][FAIL] timeout after ${TIMEOUT_SEC}s"
    kill -TERM "${cmd_pid}" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL "${cmd_pid}" >/dev/null 2>&1 || true
    wait "${cmd_pid}" >/dev/null 2>&1 || true
    tail -n 120 "${LOG_PATH}" || true
    exit 1
  fi

  if [[ ${usr1_sent} -eq 0 ]] && [[ ${elapsed} -ge ${PARSE_STALL_SEC} ]]; then
    if grep -q "\[isaaclab_env_wrapper\] parsing env cfg" "${LOG_PATH}" && ! grep -q "saved_npz=" "${LOG_PATH}"; then
      echo "[SMOKE][WARN] parse-stage stall suspected; requesting faulthandler dump (SIGUSR1)"
      docker exec "${DOCKER_NAME}" bash -lc "pkill -USR1 -f export_online_episode_isaaclab.py || true" >/dev/null 2>&1 || true
      usr1_sent=1
    fi
  fi

  sleep 5
done

wait "${cmd_pid}"
cmd_code=$?
set -e

if [[ ${cmd_code} -ne 0 ]]; then
  echo "[SMOKE][FAIL] command exit=${cmd_code}"
  tail -n 120 "${LOG_PATH}" || true
  exit 1
fi

required_markers=(
  "saved_npz="
  "steps="
  "terminal_reward="
  "episode_success="
  "render_used="
  "obs_fallback_used="
  "zero_frame_count="
)

for marker in "${required_markers[@]}"; do
  if ! grep -q "${marker}" "${LOG_PATH}"; then
    echo "[SMOKE][FAIL] missing log marker: ${marker}"
    tail -n 120 "${LOG_PATH}" || true
    exit 1
  fi
done

python - <<PY
from pathlib import Path
import numpy as np

out_npz = Path('${OUT_NPZ}')
if not out_npz.exists():
    raise SystemExit('[SMOKE][FAIL] npz not found')

data = np.load(out_npz)
required = {
    'timestep', 'state', 'base_action', 'action', 'reward', 'done', 'front_image',
    'terminal_done', 'terminal_reward', 'episode_success'
}
missing = sorted(required - set(data.files))
if missing:
    raise SystemExit(f'[SMOKE][FAIL] missing npz keys: {missing}')

frames = data['front_image']
if frames.ndim != 4:
    raise SystemExit(f'[SMOKE][FAIL] front_image ndim={frames.ndim}')

print('[SMOKE][OK] npz_schema_valid')
print(f'[SMOKE][INFO] frames_shape={frames.shape} min={int(frames.min())} max={int(frames.max())}')
print(f'[SMOKE][INFO] terminal_done={bool(data["terminal_done"])} terminal_reward={float(data["terminal_reward"]):.6f} episode_success={bool(data["episode_success"])}')
PY

echo "[SMOKE][OK] online export smoke passed"
grep -nE 'saved_npz|steps=|terminal_reward|episode_success|render_used|obs_fallback_used|zero_frame_count' "${LOG_PATH}" | tail -n 80
