#!/bin/bash
# Launch residual TD3 training on Stack Cube with difficulty levels.
#
# Usage:
#   DIFFICULTY=easy bash scripts/slurm_train_stack_td3.sh
#   DIFFICULTY=normal bash scripts/slurm_train_stack_td3.sh
#   DIFFICULTY=hard bash scripts/slurm_train_stack_td3.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DIFFICULTY="${DIFFICULTY:-easy}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
NUM_ENVS="${NUM_ENVS:-30}"
PARTITION="${PARTITION:-core}"
QOS="${QOS:-core-own}"
GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000"
EPISODE_POS="configs/stack_cube_positions.json"

# Difficulty-specific settings
# random_cube_range: [white_range, green_range]
#   dx/dy in cm, yaw in degrees
case "$DIFFICULTY" in
    easy)
        # Fixed positions from training data (no perturbation)
        RANGE_ARG=""
        OFFLINE_DIR="outputs/offline_data/stack_75k"
        ;;
    normal)
        # White cube: ±2cm XY, ±15° yaw. Green cube: fixed.
        RANGE_ARG='[{"dx":[-2,2],"dy":[-2,2],"yaw":[-15,15]}, null]'
        OFFLINE_DIR="outputs/offline_data/stack_75k"
        ;;
    hard)
        # White cube: ±4cm XY, ±30° yaw. Green cube: ±2cm XY.
        RANGE_ARG='[{"dx":[-4,4],"dy":[-4,4],"yaw":[-30,30]}, {"dx":[-2,2],"dy":[-2,2]}]'
        OFFLINE_DIR="outputs/offline_data/stack_75k"
        ;;
esac

WANDB_NAME="${WANDB_NAME:-stack_td3_75k_${DIFFICULTY}}"

echo "[Train Stack TD3] Difficulty: $DIFFICULTY"
echo "  WANDB_NAME=$WANDB_NAME, TOTAL_STEPS=$TOTAL_STEPS, NUM_ENVS=$NUM_ENVS"
echo "  RANGE: $RANGE_ARG"

sbatch \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --gres=gpu:1 \
    --cpus-per-task=14 \
    --mem=200G \
    --job-name="stk_${DIFFICULTY}" \
    --output="/home/nas_main/kinamkim/slurms/stack_train_${DIFFICULTY}_%j.out" \
    --error="/home/nas_main/kinamkim/slurms/stack_train_${DIFFICULTY}_%j.err" \
    --time=48:00:00 \
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

echo '[Train] Stack TD3 ${DIFFICULTY}'
echo '[Train] Start: \$(date)'

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_stack.py \\
    --groot_checkpoint ${GROOT_CKPT} \\
    --episode_positions_file ${EPISODE_POS} \\
    --num_envs ${NUM_ENVS} \\
    --max_episode_steps 1000 \\
    --total_timesteps ${TOTAL_STEPS} \\
    --offline_data_dir ${OFFLINE_DIR} \\
    --offline_fraction 0.5 \\
    --offline_reward_relabel computed \\
    --reward_config configs/reward_stack.yaml \\
    --reward_type dense \\
    --asymmetric_critic \\
    --gamma 0.99 \\
    --n_step 3 \\
    --batch_size 256 \\
    --buffer_size 500000 \\
    --actor_lr 3e-4 --critic_lr 3e-4 \\
    --action_scale 0.1 \\
    --action_l2_reg 1.0 \\
    --target_tau 0.005 \\
    --learning_starts 1000 \\
    --critic_warmup_steps 2000 \\
    --eval_interval 2000 \\
    --eval_num_episodes 3 \\
    --checkpoint_interval 10000 \\
    --async_eval \\
    --eval_positions_file ${EPISODE_POS} \\
    --device cuda:0 \\
    --output_dir outputs/stack_td3_75k/${DIFFICULTY} \\
    --wandb_mode online \\
    --wandb_project mujoco-franka-stack-residual-td3 \\
    --wandb_name ${WANDB_NAME} \\
    --task_description 'Pick up the white cube and stack it on the green cube.' \\
    ${RANGE_ARG:+--random_cube_range \"${RANGE_ARG}\"}

echo '[Train] Done: \$(date)'
"

echo "[Train] Job submitted. Monitor: squeue -u kinamkim"
