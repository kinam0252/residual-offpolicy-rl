#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Robustness Eval — Single checkpoint, clean + noisy eval
#
# Loads best.pt, runs eval under clean (no augmentation) and
# noisy (noise=0.01, dropout=0.1) conditions. Outputs JSON results.
#
# Usage:  TASK=pnp CKPT_DIR=outputs/pnp_rl_747 CONDITION=clean \
#           sbatch scripts/slurm/eval_robustness.sh
# ══════════════════════════════════════════════════════════════════

#SBATCH --job-name=rob_eval
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=219G
#SBATCH --time=1:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err

set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib:$HOME/lib-compat:$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export PATH=~/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

cd ~/Repos/Intern/residual-offpolicy-rl

# ── Parameters ──
TASK=${TASK:?ERROR: TASK must be set}
CKPT_DIR=${CKPT_DIR:?ERROR: CKPT_DIR must be set}
CONDITION=${CONDITION:?ERROR: CONDITION must be set (clean/noise/drop/both)}
EVAL_EPISODES=${EVAL_EPISODES:-3}
NOISE_MAX=${NOISE_MAX:-0.01}
DROPOUT_PROB=${DROPOUT_PROB:-0.1}

BEST_PT="${CKPT_DIR}/checkpoints/best.pt"
if [ ! -f "$BEST_PT" ]; then
    echo "ERROR: best.pt not found at $BEST_PT"
    exit 1
fi

OUTPUT_BASE="outputs/robustness_eval/${TASK}_${CONDITION}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_BASE"

echo "═══════════════════════════════════════════════════════"
echo " Robustness Eval: task=$TASK condition=$CONDITION"
echo " checkpoint: $BEST_PT"
echo " eval_episodes: $EVAL_EPISODES"
echo "═══════════════════════════════════════════════════════"

# ── Run eval script ──
python3 scripts/eval_robustness_standalone.py \
    --task "$TASK" \
    --checkpoint "$BEST_PT" \
    --output_dir "$OUTPUT_BASE" \
    --eval_num_episodes "$EVAL_EPISODES" \
    --noise_max "$NOISE_MAX" \
    --dropout_prob "$DROPOUT_PROB"

echo "Done. Results in $OUTPUT_BASE"
