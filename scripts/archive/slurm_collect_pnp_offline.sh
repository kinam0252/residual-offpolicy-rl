#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=pnp_offln
#SBATCH --output=/home/nas_main/kinamkim/slurms/pnp_offline_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/pnp_offline_%j.err
#SBATCH --time=6:00:00

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

DIFFICULTY="${DIFFICULTY:-easy}"
GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-100000"
EPISODE_POS="configs/pnp_sim33ep_positions.json"
OUTPUT_DIR="outputs/offline_data/pnp_${DIFFICULTY}"
RANGE_ARG=""

case "$DIFFICULTY" in
    easy)
        ;;
    normal)
        RANGE_ARG='[null, {"dx":[-6,6],"dy":[-6,6],"yaw":[-30,30]}]'
        ;;
    hard)
        RANGE_ARG='[null, {"dx":[-6,6],"dy":[-6,6],"yaw":[-30,30]}, {"dx":[-10,10],"dy":[-10,10],"yaw":[-60,60]}]'
        ;;
esac

echo "[PnP Offline] Difficulty: $DIFFICULTY"
echo "[PnP Offline] Checkpoint: $GROOT_CKPT"
echo "[PnP Offline] Output: $OUTPUT_DIR"
echo "[PnP Offline] Start: $(date)"

python3 resfit/rl_finetuning/scripts/collect_offline_data_pnp.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --episode_positions_file "$EPISODE_POS" \
    --num_episodes_per_env 300 \
    --max_episode_steps 500 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --task_description "Pick up the red cube and place it onto the plate." \
    --resume \
    ${RANGE_ARG:+--random_cube_range "$RANGE_ARG"}

echo "[PnP Offline] Done: $(date)"
