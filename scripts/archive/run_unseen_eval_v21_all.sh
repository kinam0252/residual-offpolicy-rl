#!/usr/bin/env bash
# Batch unseen eval v21: 4 concurrent, newest first, skip existing
set -uo pipefail

CKPT_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/pickMushroom_resid_v21_phase_probe/20260317_154101/checkpoints"
OUT_BASE="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/unseen_eval_v21_all"
CONTAINER="isaaclab_resfit"
PROBE="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/configs/phase_probe.pt"
CONCURRENT=4
POLL=20

STEPS=()
for f in $(ls "$CKPT_DIR"/agent_step*.pt | sort -Vr); do
    step=$(basename "$f" .pt | sed 's/agent_step//')
    outdir="$OUT_BASE/v21_step${step}_unseen"
    if [ -f "$outdir/results.json" ]; then
        echo "SKIP step $step (done)"
    else
        STEPS+=("$step")
    fi
done
echo "Remaining: ${#STEPS[@]} checkpoints"
[ ${#STEPS[@]} -eq 0 ] && echo "All done!" && exit 0

mkdir -p "$OUT_BASE"
idx=0
while [ $idx -lt ${#STEPS[@]} ]; do
    end=$((idx + CONCURRENT))
    [ $end -gt ${#STEPS[@]} ] && end=${#STEPS[@]}

    BATCH_STEPS=()
    for ((j=idx; j<end; j++)); do
        s="${STEPS[$j]}"
        BATCH_STEPS+=("$s")
        label="v21_step${s}_unseen"
        logfile="$OUT_BASE/${label}.log"
        launcher="$OUT_BASE/_launch_${s}.sh"
        cat > "$launcher" <<EOF
#!/usr/bin/env bash
set -e
export TERM=xterm PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab
exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/eval_single_run.py \
  --headless --enable_cameras \
  --checkpoint "$CKPT_DIR/agent_step${s}.pt" \
  --env_type unseen --run_label "$label" \
  --output_dir "$OUT_BASE/$label" \
  --csv_base_dir /home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758 \
  --groot_model_path /home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000 \
  --groot_embodiment_tag new_embodiment --groot_policy_device cuda:0 \
  --language_override "pick up mushroom" \
  --max_episode_steps 1000 --success_threshold 0.03 \
  --phase_probe_path "$PROBE"
EOF
        chmod +x "$launcher"
        echo "  Launch step $s"
        docker exec -d "$CONTAINER" bash -c "bash $launcher > $logfile 2>&1"
    done

    echo "Batch: ${BATCH_STEPS[*]} (${#BATCH_STEPS[@]} concurrent)"

    while true; do
        sleep $POLL
        running=$(docker exec "$CONTAINER" ps aux 2>/dev/null | grep eval_single_run | grep python3 | grep -v grep | grep -v defunct | wc -l)
        done_n=0
        for s in "${BATCH_STEPS[@]}"; do
            [ -f "$OUT_BASE/v21_step${s}_unseen/results.json" ] && done_n=$((done_n+1))
        done
        echo "  Running=$running Done=$done_n/${#BATCH_STEPS[@]}"
        [ "$running" -eq 0 ] && break
        [ "$done_n" -ge "${#BATCH_STEPS[@]}" ] && break
    done

    for s in "${BATCH_STEPS[@]}"; do
        rf="$OUT_BASE/v21_step${s}_unseen/results.json"
        if [ -f "$rf" ]; then
            sr=$(python3 -c "import json; d=json.load(open('$rf')); print(f'{d[\"success_rate\"]*100:.0f}%')" 2>/dev/null || echo "?")
            echo "  step $s => $sr"
        else
            echo "  step $s => FAILED"
        fi
    done
    idx=$end
done

echo ""
echo "=== ALL RESULTS ==="
for f in $(ls "$OUT_BASE"/v21_step*_unseen/results.json 2>/dev/null | sort -t'p' -k2 -n); do
    step=$(echo "$f" | grep -oP 'step\K[0-9]+')
    sr=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d[\"success_rate\"]*100:.0f}%')" 2>/dev/null || echo "?")
    echo "  step $step: $sr"
done
