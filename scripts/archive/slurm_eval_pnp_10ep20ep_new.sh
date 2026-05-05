#!/bin/bash
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=pnp_beval
#SBATCH --output=/home/nas_main/kinamkim/slurms/pnp_beval_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/pnp_beval_%j.err
#SBATCH --time=12:00:00

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

BASE_DIR="/home/nas_main/kinamkim/DATA/INTERN/training"
OUT_BASE="outputs/pnp_base_eval"

# 10ep and 20ep checkpoints, each with easy/normal/hard
COMBOS=(
    "groot_pnp_sim_10ep 50000 easy"
    "groot_pnp_sim_10ep 50000 normal"
    "groot_pnp_sim_10ep 50000 hard"
    "groot_pnp_sim_10ep 75000 easy"
    "groot_pnp_sim_10ep 75000 normal"
    "groot_pnp_sim_10ep 75000 hard"
    "groot_pnp_sim_20ep 50000 easy"
    "groot_pnp_sim_20ep 50000 normal"
    "groot_pnp_sim_20ep 50000 hard"
    "groot_pnp_sim_20ep 75000 easy"
    "groot_pnp_sim_20ep 75000 normal"
    "groot_pnp_sim_20ep 75000 hard"
    "groot_pnp_sim_20ep 100000 easy"
    "groot_pnp_sim_20ep 100000 normal"
    "groot_pnp_sim_20ep 100000 hard"
    "groot_pnp_sim_20ep 125000 easy"
    "groot_pnp_sim_20ep 125000 normal"
    "groot_pnp_sim_20ep 125000 hard"
)

LOCK_TIMEOUT=3600  # 1 hour stale lock timeout

echo "[Worker $$] Starting PnP base eval (10ep/20ep) on $(hostname) at $(date)"
echo "[Worker $$] Total combos: ${#COMBOS[@]}"

completed=0
while true; do
    found_work=false

    for combo in "${COMBOS[@]}"; do
        read -r ckpt_name ckpt_step difficulty <<< "$combo"

        short_name="${ckpt_name#groot_pnp_}"
        step_k=$((ckpt_step / 1000))
        out_dir="${OUT_BASE}/${short_name}_${step_k}k_${difficulty}"
        lock_dir="${out_dir}/.lock"

        # Skip if already completed
        if [ -f "${out_dir}/results.json" ]; then
            continue
        fi

        mkdir -p "$out_dir" 2>/dev/null

        # Try to acquire lock (mkdir is atomic)
        if mkdir "$lock_dir" 2>/dev/null; then
            echo "$$@$(hostname)@$(date +%s)" > "${lock_dir}/info"

            echo ""
            echo "=========================================="
            echo "[Worker $$] Running: ${short_name} step=${step_k}k ${difficulty}"
            echo "=========================================="

            ckpt_path="${BASE_DIR}/${ckpt_name}/checkpoint-${ckpt_step}"

            if [ ! -d "$ckpt_path" ]; then
                echo "[WARN] Checkpoint not found: $ckpt_path — skipping"
                rm -rf "$lock_dir"
                continue
            fi

            python3 scripts/eval_pnp_base_sr.py \
                --groot_checkpoint "$ckpt_path" \
                --difficulty "$difficulty" \
                --episodes 20 \
                --seed 42 \
                --out_dir "$out_dir" \
                --max_episode_steps 300 \
                --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
                --positions_file configs/pnp_sim33ep_positions.json

            exit_code=$?
            rm -rf "$lock_dir"

            if [ $exit_code -eq 0 ]; then
                completed=$((completed + 1))
                echo "[Worker $$] Completed ${short_name}_${step_k}k_${difficulty} (total: $completed)"
            else
                echo "[Worker $$] FAILED ${short_name}_${step_k}k_${difficulty} (exit=$exit_code)"
                rm -f "${out_dir}/results.json"
            fi

            found_work=true
            break
        else
            # Check stale lock
            lock_info="${lock_dir}/info"
            if [ -f "$lock_info" ]; then
                lock_time=$(cut -d@ -f3 < "$lock_info" 2>/dev/null || echo 0)
                now=$(date +%s)
                age=$((now - lock_time))
                if [ "$age" -gt "$LOCK_TIMEOUT" ]; then
                    echo "[Worker $$] Stale lock (age=${age}s) — removing"
                    rm -rf "$lock_dir"
                fi
            fi
            continue
        fi
    done

    if [ "$found_work" = false ]; then
        echo ""
        echo "[Worker $$] No more work available. Completed $completed combos."
        break
    fi
done

echo "[Worker $$] Done at $(date). Total completed: $completed"
