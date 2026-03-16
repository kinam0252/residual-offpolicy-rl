#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Config for v6: async eval version (train = no inline eval)
# Based on config_train.sh but with DISABLE_EVAL=true and separate EXP_NAME
# ═══════════════════════════════════════════════════════════════════

# ── Experiment name / tags ──
export EXP_NAME="pickMushroom_resid_v6_async"
export EXP_GROUP="async_eval"
export EXP_NOTES="shaped reward, offline 0.5, async eval, thresh=0.005, friction=50k"
export SEED=42

# ── W&B ──
export WANDB_API_KEY=$(cat ~/Keys/wandb_api_key | tr -d '[:space:]')
export WANDB_MODE="online"
export WANDB_ENTITY="draftrec"
export WANDB_PROJECT="Isaaclab_Franka_Pickup"
export WANDB_LOG_EVERY=10

# ── Environment ──
export NUM_TRAIN_ENVS=1
export EVAL_NUM_ENVS=10
export MAX_EPISODE_STEPS=1000
export CSV_BASE_DIR="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081744_174"

# ── GR00T base policy ──
export GROOT_MODEL_PATH="/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000"
export GROOT_EMBODIMENT_TAG="new_embodiment"
export GROOT_POLICY_DEVICE="cuda:0"
export LANGUAGE_OVERRIDE="pick up mushroom"

# ── Algorithm ──
export TOTAL_TIMESTEPS=50000
export LEARNING_STARTS=1000
export CRITIC_WARMUP_STEPS=2000
export BATCH_SIZE=64
export BUFFER_SIZE=50000
export GAMMA=0.99
export N_STEP=3
export OFFLINE_FRACTION=0.5
export OFFLINE_DATA_DIR="/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_w_reward"
export SUCCESS_THRESHOLD=0.005
export RANDOM_ACTION_NOISE_SCALE=0.0

# ── Exploration noise ──
export STDDEV_MAX=0.05
export STDDEV_MIN=0.05

# ── Network / optimiser ──
export ACTOR_LR=1e-6
export CRITIC_LR=1e-4
export ACTION_SCALE=0.05

# ── Evaluation ──
export EVAL_INTERVAL=500
export EVAL_FIRST=false
export SAVE_VIDEO=false
export DISABLE_EVAL=true                          # <-- async eval mode

# ── Checkpointing / output ──
export OUTPUT_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/${EXP_NAME}"
export CHECKPOINT_INTERVAL=500                    # match eval interval for async eval

# ── Debug flags ──
export DEBUG_ZERO_RESIDUAL=false

# ── Docker container ──
export CONTAINER="isaaclab_resfit"
export DEVICE="cuda:0"
