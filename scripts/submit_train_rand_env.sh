#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --job-name=pnp_rand
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
ACTION_SCALE="${ACTION_SCALE:-0.1}"
OFFLINE_FRACTION="${OFFLINE_FRACTION:-0.75}"
CRITIC_WARMUP="${CRITIC_WARMUP:-5000}"
N_STEP="${N_STEP:-3}"
L2_REG="${L2_REG:-5.0}"
SEED="${SEED:-42}"
TARGET_TAU="${TARGET_TAU:-0.005}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
GAMMA="${GAMMA:-0.9}"
REWARD_TYPE="${REWARD_TYPE:-dense_clipped}"
OFFLINE_RELABEL="${OFFLINE_RELABEL:-none}"

# Perturbation ranges (cm for JSON, converted to meters in training script)
# cube: ±8cm position, ±30° yaw | bowl: ±8cm position
CUBE_RANGE='{"dx":[-8,8],"dy":[-8,8],"yaw":[-30,30]}'
BOWL_RANGE='{"dx":[-8,8],"dy":[-8,8]}'

EXP_TAG="rand_cb"
if [ "$REWARD_TYPE" != "dense_clipped" ]; then
  EXP_TAG="${EXP_TAG}_${REWARD_TYPE}"
fi

echo "[PnP Random-Env] Train with random cube+bowl perturbation"
echo "[PnP Random-Env] cube_range=${CUBE_RANGE}"
echo "[PnP Random-Env] bowl_range=${BOWL_RANGE}"
echo "[PnP Random-Env] action_scale=${ACTION_SCALE}, of=${OFFLINE_FRACTION}, cw=${CRITIC_WARMUP}"
echo "[PnP Random-Env] reward_type=${REWARD_TYPE}, l2=${L2_REG}, gamma=${GAMMA}, seed=${SEED}"
echo "[PnP Random-Env] Start: $(date)"

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_pnp.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --reward_type "$REWARD_TYPE" \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr "$ACTOR_LR" --critic_lr "$CRITIC_LR" \
    --gamma "$GAMMA" \
    --target_tau "$TARGET_TAU" \
    --num_envs 15 \
    --max_episode_steps 500 \
    --action_scale "$ACTION_SCALE" \
    --random_action_noise_scale 0.05 \
    --action_l2_reg "$L2_REG" \
    --critic_warmup_steps "$CRITIC_WARMUP" \
    --n_step "$N_STEP" \
    --offline_fraction "$OFFLINE_FRACTION" \
    --offline_data_dir "outputs/offline_data/pnp_hard_v3_66ep100k" \
    --offline_reward_relabel "$OFFLINE_RELABEL" \
    --random_cube_range "$CUBE_RANGE" \
    --random_bowl_range "$BOWL_RANGE" \
    --async_eval \
    --eval_num_episodes 1 \
    --eval_positions_file "configs/pnp_eval_medium_hard.json" \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --task_description "Pick up the red cube and place it onto the plate." \
    --output_dir "outputs/pnp_train_rand/${EXP_TAG}_as${ACTION_SCALE}_l2${L2_REG}_g${GAMMA}_s${SEED}" \
    --wandb_mode online \
    --wandb_project mujoco-franka-pnp-residual-td3 \
    --wandb_name "pnp_${EXP_TAG}_as${ACTION_SCALE}_l2${L2_REG}_g${GAMMA}_s${SEED}" \
    --seed "$SEED"
