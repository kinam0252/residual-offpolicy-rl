#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=col_pnp_v3
#SBATCH --output=/home/nas_main/kinamkim/slurms/col_pnp_v3_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/col_pnp_v3_%j.err
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

DIFFICULTY=${1:-easy}
GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-200000"
EPISODE_POS="configs/pnp_common5_positions.json"
OUTPUT_DIR="outputs/offline_data/pnp_${DIFFICULTY}_v4"
RANGE_ARG=""

case "$DIFFICULTY" in
    easy)
        ;;
    normal)
        RANGE_ARG='[null, {"dx":[-3,3],"dy":[-3,3],"yaw":[-15,15]}]'
        ;;
    hard)
        RANGE_ARG='[null, {"dx":[-3,3],"dy":[-3,3],"yaw":[-15,15]}, {"dx":[-6,6],"dy":[-6,6],"yaw":[-45,45]}]'
        ;;
esac

echo "[PnP Offline v4] Difficulty: $DIFFICULTY"
echo "[PnP Offline v4] Checkpoint: $GROOT_CKPT"
echo "[PnP Offline v4] Output: $OUTPUT_DIR"

python3 resfit/rl_finetuning/scripts/collect_offline_data_pnp.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --episode_positions_file "$EPISODE_POS" \
    --num_episodes_per_env 100 \
    --max_episode_steps 500 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --ema_alpha 0.9 \
    --resume \
    ${RANGE_ARG:+--random_cube_range "$RANGE_ARG"}

echo "DONE: PnP offline data collection v4 (${DIFFICULTY})"
