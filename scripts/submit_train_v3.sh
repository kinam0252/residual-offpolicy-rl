#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --job-name=pnp_v3
#SBATCH --output=/home/nas_main/kinamkim/Repos/Intern/outputs/train_logs/slurm_%j.log
#SBATCH --error=/home/nas_main/kinamkim/Repos/Intern/outputs/train_logs/slurm_%j.err
#SBATCH --time=72:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_API_KEY="$(cat /home/nas_main/kinamkim/Keys/wandb_api_key)"
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

GROOT_CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"

# --- Configurable via env vars ---
DIFFICULTY="${DIFFICULTY:?Must set DIFFICULTY (normal or hard)}"
SEED="${SEED:-42}"
ACTION_SCALE="${ACTION_SCALE:-0.1}"
L2_REG="${L2_REG:-5.0}"
GAMMA="${GAMMA:-0.9}"
OFFLINE_FRACTION="${OFFLINE_FRACTION:-0.75}"
CRITIC_WARMUP="${CRITIC_WARMUP:-5000}"

# Perturbation ranges by difficulty (cm units, converted internally)
if [ "$DIFFICULTY" = "normal" ]; then
    CUBE_RANGE='{"dx":[-2,2],"dy":[-2,2],"yaw":[-15,15]}'
    BOWL_RANGE='{"dx":[-2,2],"dy":[-2,2]}'
    EVAL_FILE="configs/pnp_eval_normal.json"
elif [ "$DIFFICULTY" = "hard" ]; then
    CUBE_RANGE='{"dx":[-5,5],"dy":[-5,5],"yaw":[-30,30]}'
    BOWL_RANGE='{"dx":[-5,5],"dy":[-5,5]}'
    EVAL_FILE="configs/pnp_eval_hard.json"
else
    echo "ERROR: DIFFICULTY must be 'normal' or 'hard'"
    exit 1
fi

EXP_TAG="v3_${DIFFICULTY}"

echo "[PnP-v3] difficulty=${DIFFICULTY} seed=${SEED}"
echo "[PnP-v3] cube_range=${CUBE_RANGE} bowl_range=${BOWL_RANGE}"
echo "[PnP-v3] eval_file=${EVAL_FILE}"
echo "[PnP-v3] action_scale=${ACTION_SCALE}, l2=${L2_REG}, gamma=${GAMMA}, of=${OFFLINE_FRACTION}"
echo "[PnP-v3] Start: $(date)"

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_pnp.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --reward_type dense_v3 \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr 3e-4 --critic_lr 3e-4 \
    --gamma "$GAMMA" \
    --target_tau 0.005 \
    --num_envs 15 \
    --max_episode_steps 500 \
    --action_scale "$ACTION_SCALE" \
    --random_action_noise_scale 0.05 \
    --action_l2_reg "$L2_REG" \
    --critic_warmup_steps "$CRITIC_WARMUP" \
    --n_step 3 \
    --offline_fraction "$OFFLINE_FRACTION" \
    --offline_data_dir "outputs/offline_data/pnp_66ep_dense_v3" \
    --offline_reward_relabel none \
    --random_cube_range "$CUBE_RANGE" \
    --random_bowl_range "$BOWL_RANGE" \
    --async_eval \
    --eval_num_episodes 1 \
    --eval_positions_file "$EVAL_FILE" \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --task_description "Pick up the red cube and place it onto the plate." \
    --output_dir "outputs/pnp_train_v3/${EXP_TAG}_as${ACTION_SCALE}_l2${L2_REG}_g${GAMMA}_s${SEED}" \
    --wandb_mode online \
    --wandb_project mujoco-franka-pnp-residual-td3 \
    --wandb_name "pnp_${EXP_TAG}_as${ACTION_SCALE}_l2${L2_REG}_g${GAMMA}_s${SEED}" \
    --seed "$SEED"
