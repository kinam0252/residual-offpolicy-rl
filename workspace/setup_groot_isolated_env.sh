#!/usr/bin/env bash
set -euo pipefail

# Install isolated dependency set into workspace-local target directory only.
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${WORKSPACE_DIR}/.pydeps_groot_iso"

ISAACLAB_ROOT="${ISAACLAB_ROOT:-}"

if [[ -z "${ISAACLAB_ROOT}" ]]; then
  CANDIDATES=(
    "${WORKSPACE_DIR}/../../Honda_IsaacLab"
    "/home/kinam/Desktop/Repos/VLA_RL/Honda_IsaacLab"
  )

  for candidate in "${CANDIDATES[@]}"; do
    if [[ -d "${candidate}" ]]; then
      ISAACLAB_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${ISAACLAB_ROOT}" ]]; then
  echo "[ERROR] Could not resolve ISAACLAB_ROOT." >&2
  echo "[HINT] Export ISAACLAB_ROOT, e.g.:" >&2
  echo "       export ISAACLAB_ROOT=\"${WORKSPACE_DIR}/../../Honda_IsaacLab\"" >&2
  exit 1
fi

ISAACLAB_PYTHON="${ISAACLAB_ROOT}/_isaac_sim/python.sh"
if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
  echo "[ERROR] Isaac python not found: ${ISAACLAB_PYTHON}" >&2
  if [[ -L "${ISAACLAB_ROOT}/_isaac_sim" ]]; then
    echo "[HINT] ${ISAACLAB_ROOT}/_isaac_sim is a symlink. Verify its target exists." >&2
  fi
  echo "[HINT] Set a valid install root via ISAACLAB_ROOT (must contain _isaac_sim/python.sh)." >&2
  exit 1
fi

echo "[INFO] Using IsaacLab root: ${ISAACLAB_ROOT}"

mkdir -p "${DEPS_DIR}"

echo "[INFO] Installing into isolated target: ${DEPS_DIR}"
"${ISAACLAB_PYTHON}" -m pip install --upgrade pip setuptools wheel

# Use Isaac's Python ABI (3.10) and install to target directory.
# Intentionally do NOT install torch here (Isaac runtime already provides torch).
"${ISAACLAB_PYTHON}" -m pip install --target "${DEPS_DIR}" \
  tyro==0.9.17 \
  msgpack==1.1.0 \
  msgpack-numpy==0.4.8 \
  omegaconf==2.3.0 \
  termcolor==3.2.0 \
  click==8.1.8 \
  einops==0.8.1 \
  transformers==4.51.3 \
  peft==0.17.1 \
  diffusers==0.35.1 \
  datasets==3.6.0 \
  pyzmq==27.0.1 \
  pandas==2.2.3 \
  numpy==1.26.4 \
  scipy==1.15.3

echo "[DONE] Isolated dependency folder ready: ${DEPS_DIR}"
echo "[NEXT] Run: bash ${WORKSPACE_DIR}/run_franka_gr00t_iface_workspace.sh"
