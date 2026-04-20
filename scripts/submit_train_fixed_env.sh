#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=pnp_fixed
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
L2_REG="${L2_REG:-0.1}"
SEED="${SEED:-42}"
TARGET_TAU="${TARGET_TAU:-0.005}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
GAMMA="${GAMMA:-0.99}"
RC_TAG="${RC_TAG:-}"
REWARD_TYPE="${REWARD_TYPE:-dense_clipped}"
OFFLINE_RELABEL="${OFFLINE_RELABEL:-none}"

# Build experiment tag for output dir/wandb name
# Include reward type if not default, and relabel if not default
EXP_TAG="${RC_TAG}"
if [ "$REWARD_TYPE" != "dense_clipped" ]; then
  EXP_TAG="${EXP_TAG}_${REWARD_TYPE}"
fi
if [ "$OFFLINE_RELABEL" != "none" ]; then
  EXP_TAG="${EXP_TAG}_rl${OFFLINE_RELABEL}"
fi

echo "[PnP Fixed-Env] Train on eval positions (pnp_eval_hard.json, 15 fixed)"
echo "[PnP Fixed-Env] NO random perturbation"
echo "[PnP Fixed-Env] action_scale=${ACTION_SCALE}, of=${OFFLINE_FRACTION}, cw=${CRITIC_WARMUP}"
echo "[PnP Fixed-Env] reward_type=${REWARD_TYPE}, offline_relabel=${OFFLINE_RELABEL}, l2=${L2_REG}, gamma=${GAMMA}"
echo "[PnP Fixed-Env] EXP_TAG=${EXP_TAG}"
echo "[PnP Fixed-Env] Start: $(date)"

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
    --episode_positions_file "configs/pnp_eval_hard.json" \
    --async_eval \
    --eval_num_episodes 1 \
    --eval_positions_file "configs/pnp_eval_hard.json" \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --task_description "Pick up the red cube and place it onto the plate." \
    --output_dir "outputs/pnp_train_fixed_env/as${ACTION_SCALE}_of${OFFLINE_FRACTION}_cw${CRITIC_WARMUP}_ns${N_STEP}_l2${L2_REG}_g${GAMMA}${EXP_TAG}_s${SEED}" \
    --wandb_mode online \
    --wandb_project mujoco-franka-pnp-residual-td3 \
    --wandb_name "pnp_fixed_as${ACTION_SCALE}_of${OFFLINE_FRACTION}_cw${CRITIC_WARMUP}_ns${N_STEP}_l2${L2_REG}_g${GAMMA}${EXP_TAG}_s${SEED}" \
    --seed "$SEED"
