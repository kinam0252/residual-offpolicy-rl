#!/bin/bash
#SBATCH --partition=free
#SBATCH --qos=core-on-free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=pnp_fair
#SBATCH --output=/home/nas_main/kinamkim/slurms/pnp_fair_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/pnp_fair_%j.err
#SBATCH --time=12:00:00
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

BASE_DIR="/home/nas_main/kinamkim/DATA/INTERN/training"
OUT_BASE="outputs/pnp_fair_eval"
POS_FILE="configs/pnp_common5_positions.json"
NUM_RUNS=3

# All checkpoints to evaluate
COMBOS=(
    "groot_pnp_sim_10ep 25000"
    "groot_pnp_sim_10ep 50000"
    "groot_pnp_sim_10ep 75000"
    "groot_pnp_sim_20ep 25000"
    "groot_pnp_sim_20ep 50000"
    "groot_pnp_sim_20ep 75000"
    "groot_pnp_sim_20ep 100000"
    "groot_pnp_sim_20ep 125000"
    "groot_pnp_sim_33ep 100000"
    "groot_pnp_sim_33ep 200000"
    "groot_pnp_sim_33ep 300000"
    "groot_pnp_sim_66ep 100000"
    "groot_pnp_sim_66ep 200000"
    "groot_pnp_sim_66ep 300000"
    "groot_pnp_sim_100ep 100000"
    "groot_pnp_sim_100ep 200000"
    "groot_pnp_sim_100ep 300000"
)

LOCK_TIMEOUT=3600

echo "[Worker $$] Starting FAIR eval (common 5 pos x2, 3 runs, easy) on $(hostname) at $(date)"
echo "[Worker $$] Total combos: ${#COMBOS[@]}, runs per combo: $NUM_RUNS"

completed=0
while true; do
    found_work=false

    for combo in "${COMBOS[@]}"; do
        read -r ckpt_name ckpt_step <<< "$combo"

        short_name="${ckpt_name#groot_pnp_}"
        step_k=$((ckpt_step / 1000))
        out_dir="${OUT_BASE}/${short_name}_${step_k}k"
        lock_dir="${out_dir}/.lock"

        if [ -f "${out_dir}/fair_results.json" ]; then
            continue
        fi

        mkdir -p "$out_dir" 2>/dev/null

        if mkdir "$lock_dir" 2>/dev/null; then
            echo "$$@$(hostname)@$(date +%s)" > "${lock_dir}/info"

            echo ""
            echo "=========================================="
            echo "[Worker $$] Running: ${short_name} step=${step_k}k ($NUM_RUNS runs)"
            echo "=========================================="

            ckpt_path="${BASE_DIR}/${ckpt_name}/checkpoint-${ckpt_step}"

            if [ ! -d "$ckpt_path" ]; then
                echo "[WARN] Checkpoint not found: $ckpt_path — skipping"
                rm -rf "$lock_dir"
                continue
            fi

            for run_i in $(seq 1 $NUM_RUNS); do
                run_dir="${out_dir}/run_${run_i}"
                
                python3 scripts/eval_pnp_base_sr.py \
                    --groot_checkpoint "$ckpt_path" \
                    --difficulty easy \
                    --episodes 10 \
                    --out_dir "$run_dir" \
                    --max_episode_steps 300 \
                    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
                    --positions_file "$POS_FILE" \
                    --no_video

                if [ $? -eq 0 ] && [ -f "${run_dir}/results.json" ]; then
                    sr=$(python3 -c "import json; print(json.load(open('${run_dir}/results.json'))['success_rate'])")
                    echo "  run $run_i: SR=$sr"
                else
                    echo "  run $run_i: FAILED"
                fi
            done

            # Aggregate results from run dirs
            python3 -c "
import json, os, glob
run_dirs = sorted(glob.glob('${out_dir}/run_*'))
srs = []
for rd in run_dirs:
    rf = os.path.join(rd, 'results.json')
    if os.path.exists(rf):
        srs.append(json.load(open(rf))['success_rate'])
if srs:
    avg = sum(srs) / len(srs)
    result = {'avg_sr': avg, 'run_srs': srs, 'num_runs': len(srs), 'positions': 'common5x2', 'difficulty': 'easy'}
    json.dump(result, open('${out_dir}/fair_results.json', 'w'), indent=2)
    print(f'  AVG SR = {avg*100:.1f}% (runs: {[\"{:.0f}%\".format(s*100) for s in srs]})')
else:
    print('  NO valid run results!')
"

            rm -rf "$lock_dir"
            completed=$((completed + 1))
            echo "[Worker $$] Completed ${short_name}_${step_k}k (total: $completed)"

            found_work=true
            break
        else
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
