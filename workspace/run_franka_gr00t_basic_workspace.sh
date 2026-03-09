#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DEPRECATED] run_franka_gr00t_basic_workspace.sh now forwards to iface launcher."
echo "[DEPRECATED] Use: bash ${WORKSPACE_DIR}/run_franka_gr00t_iface_workspace.sh"

exec bash "${WORKSPACE_DIR}/run_franka_gr00t_iface_workspace.sh" "$@"
