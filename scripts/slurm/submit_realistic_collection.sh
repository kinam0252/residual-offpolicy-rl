#!/bin/bash
# submit_realistic_collection.sh — Submit parallel realistic data collection jobs
#
# Usage:
#   bash scripts/slurm/submit_realistic_collection.sh          # submit all 3 tasks
#   bash scripts/slurm/submit_realistic_collection.sh pnp      # submit pnp only
#   bash scripts/slurm/submit_realistic_collection.sh stack 5   # stack with 5 workers
#
# Safe to re-run: lock-based collection skips completed/in-progress episodes.
# Safe to cancel: stale locks are reclaimed after 30 minutes.

set -euo pipefail

JOBS_PER_TASK="${2:-3}"  # default 3 parallel workers per task
QOS="${3:-extra}"       # default qos=extra, can override (e.g., share)

# Task configurations: TASK POSITIONS_FILE EPISODES_PER_POS GROOT_CHECKPOINT OUTPUT_DIR
declare -A TASK_CFG
TASK_CFG[pnp]="configs/pnp_train_30.json|10|~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000|outputs/offline_data/pnp_realistic"
TASK_CFG[stack]="configs/stack_train_30.json|10|~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000|outputs/offline_data/stack_realistic"
TASK_CFG[lift]="configs/lift_66ep_positions.json|3|~/DATA/INTERN/training/groot_lift_sim_32ep_100k/checkpoint-100000|outputs/offline_data/lift_realistic"

# Which tasks to submit
if [ $# -ge 1 ] && [ "${1}" != "all" ]; then
    TASKS=("${1}")
else
    TASKS=(pnp stack lift)
fi

echo "=== Realistic Data Collection ==="
echo "Tasks: ${TASKS[*]}"
echo "Workers per task: ${JOBS_PER_TASK}"
echo "QOS: ${QOS}"
echo ""

for TASK in "${TASKS[@]}"; do
    IFS='|' read -r POSITIONS EPS_PER_POS GROOT_CKPT OUTPUT_DIR <<< "${TASK_CFG[$TASK]}"

    echo "--- ${TASK} ---"
    echo "  Positions: ${POSITIONS}"
    echo "  Episodes/pos: ${EPS_PER_POS}"
    echo "  GR00T: ${GROOT_CKPT}"
    echo "  Output: ${OUTPUT_DIR}"

    for i in $(seq 1 "${JOBS_PER_TASK}"); do
        # Write a temp script (--wrap has /bin/sh issues with bash syntax)
        SCRIPT=$(mktemp /tmp/real_${TASK}_XXXXXX.sh)
        cat > "${SCRIPT}" <<INNER_EOF
#!/bin/bash
set -eu
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=\${CUDA_VISIBLE_DEVICES%%,*}
NV=\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=\$NV/cu13/lib:\$NV/cuda_runtime/lib:\$NV/cublas/lib:\$NV/cudnn/lib:\$NV/cufft/lib:\$NV/cusolver/lib:\$NV/cusparse/lib:\$NV/nvjitlink/lib:\$NV/cuda_nvrtc/lib:\$NV/nccl/lib:\$HOME/.local/lib:\$HOME/lib-compat:\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PYTHONPATH="\$(pwd):\${PYTHONPATH:-}"

echo "[worker ${i}/${JOBS_PER_TASK}] Task=${TASK} Job=\${SLURM_JOB_ID}"

python resfit/rl_finetuning/scripts/collect_offline_data_with_images.py \\
    --task ${TASK} \\
    --episode_positions_file ${POSITIONS} \\
    --num_episodes_per_env ${EPS_PER_POS} \\
    --max_episode_steps 300 \\
    --output_dir ${OUTPUT_DIR} \\
    --rl_img_size 84 \\
    --realistic \\
    --groot_checkpoint ${GROOT_CKPT}
INNER_EOF
        chmod +x "${SCRIPT}"

        JOB_ID=$(sbatch \
            --job-name="real_${TASK}_${i}" \
            --output="/home/nas_main/kinamkim/slurms/real_${TASK}_%j.out" \
            --error="/home/nas_main/kinamkim/slurms/real_${TASK}_%j.err" \
            --qos=${QOS} \
            --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 \
            --time=24:00:00 \
            "${SCRIPT}" 2>&1 | grep -oP '\d+')
        echo "  Worker ${i}: Job ${JOB_ID}"
    done
    echo ""
done

echo "=== All jobs submitted ==="
echo "Monitor: squeue -u \$(whoami) -n real_pnp,real_stack,real_lift"
echo "Progress: ls outputs/offline_data/{pnp,stack,lift}_realistic/ep_*.npz | wc -l"
