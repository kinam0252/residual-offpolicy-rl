#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --job-name=stk_rl
#SBATCH --output=/home/nas_main/kinamkim/slurms/stack_rl_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/stack_rl_%j.err
#SBATCH --time=72:00:00

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

# --- Configuration ---
DIFFICULTY="${DIFFICULTY:-easy}"
GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000"
EPISODE_POS="configs/stack_cube_positions.json"
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

echo "[Stack RL] Difficulty: $DIFFICULTY"
echo "[Stack RL] Checkpoint: $GROOT_CKPT"
echo "[Stack RL] Start: $(date)"

# Wandb API key
if [ -f ~/Keys/wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/Keys/wandb_api_key)
fi

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_stack.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --episode_positions_file "$EPISODE_POS" \
    --reward_type dense \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr 3e-4 --critic_lr 3e-4 \
    --gamma 0.95 \
    --target_tau 0.005 \
    --num_envs 30 \
    --max_episode_steps 500 \
    --action_scale 0.1 \
    --random_action_noise_scale 0.05 \
    --action_l2_reg 1.0 \
    --critic_warmup_steps 2000 \
    --n_step 3 \
    --offline_fraction 0.75 \
    --offline_data_dir "outputs/offline_data/stack_${DIFFICULTY}" \
    --async_eval \
    --eval_num_episodes 5 \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --task_description "Pick up the white cube and stack it on the green cube." \
    --output_dir "outputs/stack_rl/${DIFFICULTY}" \
    --wandb_mode online \
    --wandb_project mujoco-franka-stack-residual-td3 \
    --wandb_name "stack_66ep100k_${DIFFICULTY}" \
    ${RANGE_ARG:+--random_cube_range "$RANGE_ARG"}

echo "[Stack RL] Done: $(date)"
