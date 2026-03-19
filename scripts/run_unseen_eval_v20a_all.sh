#!/usr/bin/env bash
set -uo pipefail
CKPT_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/pickMushroom_resid_v20a_vlm_fix_only/20260317_122008/checkpoints"
OUT_BASE="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/unseen_eval_v20a_all"
CONTAINER="isaaclab_resfit"
CONCURRENT=4
POLL=20

STEPS=()
for f in $(ls "$CKPT_DIR"/agent_step*.pt | sort -Vr); do
    step=$(basename "$f" .pt | sed 's/agent_step//')
    if [ -f "$OUT_BASE/v20a_step${step}_unseen/results.json" ]; then
        echo "SKIP step $step"
    else
        STEPS+=("$step")
    fi
done
echo "Remaining: ${#STEPS[@]}"
[ ${#STEPS[@]} -eq 0 ] && echo "Done!" && exit 0
mkdir -p "$OUT_BASE"
idx=0
while [ $idx -lt ${#STEPS[@]} ]; do
    end=$((idx + CONCURRENT)); [ $end -gt ${#STEPS[@]} ] && end=${#STEPS[@]}
    BATCH=()
    for ((j=idx; j<end; j++)); do
        s="${STEPS[$j]}"; BATCH+=("$s")
        cat > "$OUT_BASE/_launch_${s}.sh" <<EOF
#!/usr/bin/env bash
set -e
export TERM=xterm PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl:/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T"
cd /workspace/isaaclab
exec ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/eval_single_run.py \
  --headless --enable_cameras \
  --checkpoint $CKPT_DIR/agent_step${s}.pt \
  --env_type unseen --run_label v20a_step${s}_unseen \
  --output_dir $OUT_BASE/v20a_step${s}_unseen \
  --csv_base_dir /home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758 \
  --groot_model_path /home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000 \
  --groot_embodiment_tag new_embodiment --groot_policy_device cuda:0 \
  --language_override "pick up mushroom" \
  --max_episode_steps 1000 --success_threshold 0.03
EOF
        chmod +x "$OUT_BASE/_launch_${s}.sh"
        echo "  Launch step $s"
        docker exec -d "$CONTAINER" bash -c "bash $OUT_BASE/_launch_${s}.sh > $OUT_BASE/v20a_step${s}_unseen.log 2>&1"
    done
    echo "Batch: ${BATCH[*]}"
    while true; do
        sleep $POLL
        running=$(docker exec "$CONTAINER" ps aux 2>/dev/null | grep eval_single_run | grep python3 | grep -v grep | grep -v defunct | wc -l)
        done_n=0
        for s in "${BATCH[@]}"; do [ -f "$OUT_BASE/v20a_step${s}_unseen/results.json" ] && done_n=$((done_n+1)); done
        echo "  Running=$running Done=$done_n/${#BATCH[@]}"
        [ "$running" -eq 0 ] && break; [ "$done_n" -ge "${#BATCH[@]}" ] && break
    done
    for s in "${BATCH[@]}"; do
        rf="$OUT_BASE/v20a_step${s}_unseen/results.json"
        [ -f "$rf" ] && sr=$(python3 -c "import json; print(f'{json.load(open(\"$rf\"))[\"success_rate\"]*100:.0f}%')" 2>/dev/null) && echo "  step $s => $sr" || echo "  step $s => FAILED"
    done
    idx=$end
done
echo "=== ALL ==="
for f in $(ls "$OUT_BASE"/v20a_step*_unseen/results.json 2>/dev/null | sort -t'p' -k2 -n); do
    step=$(echo "$f" | grep -oP 'step\K[0-9]+')
    sr=$(python3 -c "import json; print(f'{json.load(open(\"$f\"))[\"success_rate\"]*100:.0f}%')" 2>/dev/null)
    echo "  step $step: $sr"
done
