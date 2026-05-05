#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --job-name=pnp_rl_v2
#SBATCH --output=/home/nas_main/kinamkim/slurms/pnp_rl_v2_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/pnp_rl_v2_%j.err
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
GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-100000"

# Difficulty-specific settings
case "$DIFFICULTY" in
    easy)
        EPISODE_POS="configs/pnp_sim33ep_positions.json"
        EXTRA_ARGS="--episode_positions_file $EPISODE_POS"
        RANGE_ARG=""
        ;;
    normal)
        EPISODE_POS="configs/pnp_sim33ep_positions.json"
        EXTRA_ARGS="--episode_positions_file $EPISODE_POS"
        RANGE_ARG='[null, {"dx":[-6,6],"dy":[-6,6],"yaw":[-30,30]}]'
        ;;
    hard)
        EPISODE_POS="configs/pnp_sim33ep_positions.json"
        EXTRA_ARGS="--episode_positions_file $EPISODE_POS"
        RANGE_ARG='[null, {"dx":[-6,6],"dy":[-6,6],"yaw":[-30,30]}, {"dx":[-10,10],"dy":[-10,10],"yaw":[-60,60]}]'
        ;;
esac

echo "[PnP RL v2] Difficulty: $DIFFICULTY"
echo "[PnP RL v2] Checkpoint: $GROOT_CKPT"
echo "[PnP RL v2] Start: $(date)"

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_pnp.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --reward_type dense_v2 \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr 3e-4 --critic_lr 3e-4 \
    --gamma 0.99 \
    --target_tau 0.005 \
    --num_envs 30 \
    --max_episode_steps 500 \
    --action_scale 1.0 \
    --random_action_noise_scale 0.05 \
    --action_l2_reg 0.01 \
    --critic_warmup_steps 2000 \
    --n_step 3 \
    --offline_fraction 0.5 \
    --offline_data_dir "outputs/offline_data/pnp_${DIFFICULTY}" \
    --async_eval \
    --eval_num_episodes 5 \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --task_description "Pick up the red cube and place it onto the plate." \
    --output_dir "outputs/pnp_rl_v2_dv2_s10/${DIFFICULTY}" \
    --wandb_mode online \
    --wandb_project mujoco-franka-pnp-residual-td3 \
    --wandb_name "pnp_v2_dv2_s10_33ep100k_${DIFFICULTY}" \
    $EXTRA_ARGS \
    ${RANGE_ARG:+--random_cube_range "$RANGE_ARG"}
