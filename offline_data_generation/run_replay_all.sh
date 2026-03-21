#!/usr/bin/env bash
# Launch parallel Isaac Sim workers to replay all CSV episodes with shaped reward.
# Output: LeRobot format at OUTPUT_DIR.
set -uo pipefail

CONTAINER="isaaclab_resfit"
N_WORKERS="${N_WORKERS:-8}"
CSV_BASE="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081744_174"
OUTPUT_DIR="${OUTPUT_DIR:-/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_w_reward}"
WORKER_SCRIPT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/replay_all_worker.py"
RETRY_JSON="${RETRY_JSON:-}"  # set externally for retry mode
FRICTION="${FRICTION:-50000}"
SUCCESS_THRESH="${SUCCESS_THRESH:-0.005}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Parallel CSV Replay with Shaped Reward (${N_WORKERS} workers)"
echo "║  Output: ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"

# Clean output dir ONLY if CLEAN=1 is set
if [[ "${CLEAN:-0}" == "1" ]]; then
  echo "⚠️  CLEAN=1: Deleting existing data in ${OUTPUT_DIR}"
  docker exec "${CONTAINER}" rm -rf "${OUTPUT_DIR}/data" "${OUTPUT_DIR}/videos" 2>/dev/null || true
else
  echo "ℹ️  Keeping existing data (use CLEAN=1 to delete)"
fi
docker exec "${CONTAINER}" mkdir -p "${OUTPUT_DIR}" 2>/dev/null || true
mkdir -p "${OUTPUT_DIR}" 2>/dev/null || true

# Kill existing workers
docker exec "${CONTAINER}" pkill -9 -f replay_all_worker 2>/dev/null || true
sleep 2

# Track PIDs
> /tmp/replay_worker_pids.txt

for WID in $(seq 0 $((N_WORKERS - 1))); do
  LAUNCHER="${OUTPUT_DIR}/_worker_${WID}.sh"
  cat > "${LAUNCHER}" << EOF
#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab
exec ./isaaclab.sh -p ${WORKER_SCRIPT} \\
  --headless --enable_cameras \\
  --num_envs 1 \\
  --csv_base_dir "${CSV_BASE}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --worker_id ${WID} \\
  --num_workers ${N_WORKERS} \\
  --friction ${FRICTION} \\
  --success_threshold ${SUCCESS_THRESH} \\
  --reward_type ${REWARD_TYPE:-dense} \\
  ${RETRY_JSON:+--retry_episodes_json ${RETRY_JSON}}
EOF
  chmod +x "${LAUNCHER}"

  LOG_FILE="${OUTPUT_DIR}/worker_${WID}.log"
  docker exec -d "${CONTAINER}" bash -c "bash '${LAUNCHER}' > '${LOG_FILE}' 2>&1"
  DPID=$(docker exec "${CONTAINER}" bash -c "pgrep -f 'worker_id ${WID}' | tail -1" 2>/dev/null || echo "?")
  echo "${DPID}" >> /tmp/replay_worker_pids.txt
  echo "  Worker ${WID} launched, log: ${LOG_FILE}"
done

echo ""
echo "All ${N_WORKERS} workers launched."
echo "Monitor: tail -f ${OUTPUT_DIR}/worker_0.log"
echo "Check all: for i in \$(seq 0 $((N_WORKERS - 1))); do echo \"=== W\$i ===\"; tail -3 ${OUTPUT_DIR}/worker_\${i}.log 2>/dev/null; done"
echo "Kill all:  docker exec ${CONTAINER} pkill -9 -f replay_all_worker"
