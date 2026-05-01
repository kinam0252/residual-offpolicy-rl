#!/bin/bash
# Hard difficulty sweep: action_scale + l2_reg experiments
# H1: scale_up (0.3/0.1), H2: scale_explore (0.5/0.01), H3: aggressive (1.0/0.01)
#
# Usage:
#   VARIANT=h1 bash scripts/slurm_train_stack_td3_hard_sweep.sh
#   VARIANT=h2 bash scripts/slurm_train_stack_td3_hard_sweep.sh
#   VARIANT=h3 bash scripts/slurm_train_stack_td3_hard_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

VARIANT="${VARIANT:-h1}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
NUM_ENVS="${NUM_ENVS:-30}"
PARTITION="${PARTITION:-core}"
QOS="${QOS:-core-own}"
GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000"
EPISODE_POS="configs/stack_cube_positions.json"

# Hard perturbation: white ±4cm XY ±30° yaw, green ±2cm XY
RANGE_ARG='[{"dx":[-4,4],"dy":[-4,4],"yaw":[-30,30]}, {"dx":[-2,2],"dy":[-2,2]}]'
OFFLINE_DIR="outputs/offline_data/stack_75k"

case "$VARIANT" in
    h1)
        ACTION_SCALE=0.3
        ACTION_L2_REG=0.1
        ACTOR_LR=3e-4
        STDDEV="0.05"
        DESC="scale_up"
        ;;
    h2)
        ACTION_SCALE=0.5
        ACTION_L2_REG=0.01
        ACTOR_LR=3e-4
        STDDEV="0.1"
        DESC="scale_explore"
        ;;
    h3)
        ACTION_SCALE=1.0
        ACTION_L2_REG=0.01
        ACTOR_LR=1e-4
        STDDEV="0.1"
        DESC="aggressive"
        ;;
esac

WANDB_NAME="stack_td3_75k_hard_${DESC}"
OUTPUT_DIR="outputs/stack_td3_75k/hard_${DESC}"

echo "[Hard Sweep] Variant: $VARIANT ($DESC)"
echo "  action_scale=$ACTION_SCALE, l2_reg=$ACTION_L2_REG, actor_lr=$ACTOR_LR, stddev=$STDDEV"
echo "  output=$OUTPUT_DIR"

sbatch \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --gres=gpu:1 \
    --cpus-per-task=14 \
    --mem=200G \
    --job-name="stk_${DESC}" \
    --output="/home/nas_main/kinamkim/slurms/stack_hard_${DESC}_%j.out" \
    --error="/home/nas_main/kinamkim/slurms/stack_hard_${DESC}_%j.err" \
    --time=48:00:00 \
    ${NODELIST:+--nodelist=${NODELIST}} \
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

echo '[Train] Stack TD3 hard_${DESC}'
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
    --actor_lr ${ACTOR_LR} --critic_lr 3e-4 \\
    --action_scale ${ACTION_SCALE} \\
    --action_l2_reg ${ACTION_L2_REG} \\
    --stddev_min ${STDDEV} --stddev_max ${STDDEV} \\
    --target_tau 0.005 \\
    --learning_starts 1000 \\
    --critic_warmup_steps 2000 \\
    --eval_interval 2000 \\
    --eval_num_episodes 3 \\
    --checkpoint_interval 10000 \\
    --async_eval \\
    --eval_positions_file ${EPISODE_POS} \\
    --device cuda:0 \\
    --output_dir ${OUTPUT_DIR} \\
    --wandb_mode online \\
    --wandb_project mujoco-franka-stack-residual-td3 \\
    --wandb_name ${WANDB_NAME} \\
    --task_description 'Pick up the white cube and stack it on the green cube.' \\
    --random_cube_range '${RANGE_ARG}'

echo '[Train] Done: \$(date)'
"

echo "[Hard Sweep] Job submitted. Monitor: squeue -u kinamkim"
