#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --job-name=pnp_train_hard
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
ACTION_SCALE="${ACTION_SCALE:-1.0}"
OFFLINE_FRACTION="${OFFLINE_FRACTION:-0.5}"
CRITIC_WARMUP="${CRITIC_WARMUP:-2000}"
N_STEP="${N_STEP:-3}"
L2_REG="${L2_REG:-0.01}"

# Hard range for training env: [null(=easy), normal, hard]
RANGE_ARG='[null, {"dx":[-3,3],"dy":[-3,3],"yaw":[-15,15]}, {"dx":[-6,6],"dy":[-6,6],"yaw":[-45,45]}]'

echo "[PnP Train Hard] GR00T: 66ep/100k, action_scale=${ACTION_SCALE}"
echo "[PnP Train Hard] Eval: pnp_eval_hard.json (15 fixed envs)"
echo "[PnP Train Hard] Offline: pnp_hard_66ep100k (500 eps, 27k transitions)"
echo "[PnP Train Hard] Start: $(date)"

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_pnp.py \
    --groot_checkpoint "$GROOT_CKPT" \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --reward_type dense_clipped \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr 3e-4 --critic_lr 3e-4 \
    --gamma 0.99 \
    --target_tau 0.005 \
    --num_envs 15 \
    --max_episode_steps 500 \
    --action_scale "$ACTION_SCALE" \
    --random_action_noise_scale 0.05 \
    --action_l2_reg "$L2_REG" \
    --critic_warmup_steps "$CRITIC_WARMUP" \
    --n_step "$N_STEP" \
    --offline_fraction "$OFFLINE_FRACTION" \
    --offline_data_dir "outputs/offline_data/pnp_hard_66ep100k" \
    --episode_positions_file "configs/pnp_common5_positions.json" \
    --random_cube_range "$RANGE_ARG" \
    --async_eval \
    --eval_num_episodes 1 \
    --eval_positions_file "configs/pnp_eval_hard.json" \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --task_description "Pick up the red cube and place it onto the plate." \
    --output_dir "outputs/pnp_train_hard_66ep100k/as${ACTION_SCALE}_of${OFFLINE_FRACTION}_cw${CRITIC_WARMUP}_ns${N_STEP}_l2${L2_REG}" \
    --wandb_mode online \
    --wandb_project mujoco-franka-pnp-residual-td3 \
    --wandb_name "pnp_hard_66ep100k_as${ACTION_SCALE}_of${OFFLINE_FRACTION}_cw${CRITIC_WARMUP}_ns${N_STEP}_l2${L2_REG}"
