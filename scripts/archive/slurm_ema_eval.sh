#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=ema_eval
#SBATCH --output=/home/nas_main/kinamkim/slurms/ema_eval_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/ema_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --exclude=worker-6

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-100000"
POS_FILE="configs/pnp_common5_positions.json"
OUT_BASE="outputs/pnp_ema_eval_500"
EMAS=(0.5 0.6 0.7 0.8 0.9)
LOCK_TIMEOUT=1800

echo "[Worker $$] EMA eval on $(hostname) at $(date)"

completed=0
while true; do
    found_work=false
    for ema in "${EMAS[@]}"; do
        out_dir="${OUT_BASE}/ema_${ema}"
        lock_dir="${out_dir}/.lock"

        if [ -f "${out_dir}/results.json" ]; then
            continue
        fi

        mkdir -p "$out_dir" 2>/dev/null

        if mkdir "$lock_dir" 2>/dev/null; then
            echo "$$@$(hostname)@$(date +%s)" > "${lock_dir}/info"
            echo ""
            echo "=========================================="
            echo "[Worker $$] Running EMA=${ema}"
            echo "=========================================="

            python3 scripts/eval_pnp_base_sr.py \
                --groot_checkpoint "$CKPT" \
                --difficulty easy \
                --episodes 5 \
                --out_dir "$out_dir" \
                --max_episode_steps 500 \
                --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
                --positions_file "$POS_FILE" \
                --ema_alpha "$ema"

            rm -rf "$lock_dir"
            completed=$((completed + 1))
            echo "[Worker $$] Completed EMA=${ema} (total: $completed)"
            found_work=true
            break
        else
            lock_info="${lock_dir}/info"
            if [ -f "$lock_info" ]; then
                lock_time=$(cut -d@ -f3 < "$lock_info" 2>/dev/null || echo 0)
                now=$(date +%s)
                age=$((now - lock_time))
                if [ "$age" -gt "$LOCK_TIMEOUT" ]; then
                    rm -rf "$lock_dir"
                fi
            fi
            continue
        fi
    done

    if [ "$found_work" = false ]; then
        echo "[Worker $$] No more work. Completed $completed."
        break
    fi
done

echo "[Worker $$] Done at $(date)"
