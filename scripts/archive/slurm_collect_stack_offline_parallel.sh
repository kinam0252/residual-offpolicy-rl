#!/bin/bash
# Submit 7 parallel offline collection jobs for Stack Cube.
# Each job runs the same script on the same output_dir;
# work-stealing via .lock directories ensures no duplicated episodes.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DIFFICULTY="${DIFFICULTY:-easy}"
N_JOBS="${N_JOBS:-8}"

echo "[Stack Parallel Offline] Submitting $N_JOBS jobs for difficulty=$DIFFICULTY"

for i in $(seq 1 $N_JOBS); do
    JOB_ID=$(sbatch \
        --partition=core \
        --qos=core-own \
        --gres=gpu:1 \
        --cpus-per-task=18 \
        --mem=200G \
        --job-name="stk_of_${i}" \
        --output="/home/nas_main/kinamkim/slurms/stack_offline_${DIFFICULTY}_w${i}_%j.out" \
        --error="/home/nas_main/kinamkim/slurms/stack_offline_${DIFFICULTY}_w${i}_%j.err" \
        --time=12:00:00 \
        --wrap="
set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:\${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ${REPO_DIR}

echo \"[Worker ${i}/${N_JOBS}] Stack offline collection, difficulty=${DIFFICULTY}\"
echo \"[Worker ${i}] Start: \$(date)\"

python3 resfit/rl_finetuning/scripts/collect_offline_data_stack.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000 \
    --episode_positions_file configs/stack_cube_positions.json \
    --num_episodes_per_env 300 \
    --max_episode_steps 1000 \
    --output_dir outputs/offline_data/stack_75k \
    --device cuda:0 \
    --task_description 'Pick up the white cube and stack it on the green cube.' \
    --resume

echo \"[Worker ${i}] Done: \$(date)\"
" 2>&1)
    echo "  Worker $i: $JOB_ID"
done

echo "[Stack Parallel Offline] All $N_JOBS jobs submitted."
echo "Monitor: squeue -u kinamkim"
echo "Output:  outputs/offline_data/stack_${DIFFICULTY}/"
