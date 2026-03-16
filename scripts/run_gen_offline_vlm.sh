#!/usr/bin/env bash
# Generate offline data with GR00T VLM latent extraction
# 4 workers (each loads GR00T ~5GB + Isaac Sim ~10GB = ~15GB × 4 = 60GB)
set -uo pipefail

export N_WORKERS=4
export OUTPUT_DIR="/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_dense_clipped_3cm_vlm"
export REWARD_TYPE="dense_clipped"
export SUCCESS_THRESH="0.03"
export FRICTION="50000"
export GROOT_MODEL="/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000"

CONTAINER="isaaclab_resfit"
CSV_BASE="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758"
WORKER_SCRIPT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/replay_all_worker_with_groot.py"
PYPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"

echo "================================================================"
echo "  Generating offline data: dense_clipped + VLM latent"
echo "  Workers: ${N_WORKERS}"
echo "  Output:  ${OUTPUT_DIR}"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}" 2>/dev/null || true
docker exec "${CONTAINER}" mkdir -p "${OUTPUT_DIR}" 2>/dev/null || true

# Kill existing workers
docker exec "${CONTAINER}" pkill -9 -f replay_all_worker_with_groot 2>/dev/null || true
sleep 2

LAUNCH_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/scripts"

for WID in $(seq 0 $((N_WORKERS - 1))); do
  LAUNCHER="${LAUNCH_DIR}/_vlm_worker_${WID}.sh"
  cat > "${LAUNCHER}" << EOF
#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYPATH}"
cd /workspace/isaaclab
exec ./isaaclab.sh -p ${WORKER_SCRIPT} \\
  --headless --enable_cameras \\
  --num_envs 1 \\
  --csv_base_dir "${CSV_BASE}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --groot_model_path "${GROOT_MODEL}" \\
  --language_override "pick up mushroom" \\
  --worker_id ${WID} \\
  --num_workers ${N_WORKERS} \\
  --friction ${FRICTION} \\
  --success_threshold ${SUCCESS_THRESH} \\
  --reward_type ${REWARD_TYPE} \\
  --vlm_inference_interval 16
EOF
  chmod +x "${LAUNCHER}"

  LOG_FILE="${OUTPUT_DIR}/worker_${WID}.log"
  docker exec -d "${CONTAINER}" bash -c "bash '${LAUNCHER}' > '${LOG_FILE}' 2>&1"
  echo "  Worker ${WID} launched, log: ${LOG_FILE}"
done

echo ""
echo "All ${N_WORKERS} workers launched."
echo "Monitor: for i in \$(seq 0 $((N_WORKERS - 1))); do echo \"=== W\$i ===\"; tail -3 ${OUTPUT_DIR}/worker_\${i}.log 2>/dev/null; done"
echo "Kill all: docker exec ${CONTAINER} pkill -9 -f replay_all_worker_with_groot"
