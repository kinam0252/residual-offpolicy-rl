#!/usr/bin/env bash
# Kill only our tracked resfit processes, then run replay script
set -euo pipefail

CONTAINER="isaaclab_resfit"
CSV_DIR="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081744_174"
OUTPUT_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/offline_replay_output"
SCRIPT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/replay_csv_with_reward.py"

# ── Kill only our tracked PIDs ──
if [[ -f /tmp/resfit_pid.txt ]]; then
  OLD_PID=$(cat /tmp/resfit_pid.txt)
  echo "Killing tracked resfit PID: $OLD_PID"
  kill -9 "$OLD_PID" 2>/dev/null || true
fi
if [[ -f /tmp/resfit_docker_pid.txt ]]; then
  OLD_DPID=$(cat /tmp/resfit_docker_pid.txt)
  echo "Killing tracked docker PID: $OLD_DPID"
  kill -9 "$OLD_DPID" 2>/dev/null || true
fi
sleep 2

# ── Write launcher to bind-mounted path ──
LAUNCHER="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/_replay_launch.sh"
cat > "${LAUNCHER}" << 'EOF'
#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab
EOF

cat >> "${LAUNCHER}" << EOFVARS
exec ./isaaclab.sh -p ${SCRIPT} \\
  --headless --enable_cameras \\
  --num_envs 1 \\
  --csv_dir "${CSV_DIR}" \\
  --output_dir "${OUTPUT_DIR}"
EOFVARS
chmod +x "${LAUNCHER}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Replay CSV with shaped reward                          ║"
echo "║  CSV: ${CSV_DIR##*/}                                    "
echo "║  Output: ${OUTPUT_DIR}                                  "
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Run foreground ──
docker exec "${CONTAINER}" bash "${LAUNCHER}" 2>&1 | tee /tmp/replay_csv.log
echo ""
echo "Done. Check: ${OUTPUT_DIR}"
