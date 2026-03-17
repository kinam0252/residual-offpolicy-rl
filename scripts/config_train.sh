#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Residual TD3 on IsaacLab — Configurable Hyperparameters
# ═══════════════════════════════════════════════════════════════════
# Edit this file, then run:   bash scripts/run_train.sh
# ═══════════════════════════════════════════════════════════════════

# ── Experiment name / tags ──
export EXP_NAME="pickMushroom_resid_v19_resume_random"   # wandb run name
export EXP_GROUP="multi_env"                       # wandb group
export EXP_NOTES="v19: resume from v17 best, random cube ±5cm/±10deg, VLM, L2=10, 200K"
export SEED=42

# ── W&B ──
export WANDB_API_KEY=$(cat ~/Keys/wandb_api_key | tr -d '[:space:]')
export WANDB_MODE="online"                        # online / offline / disabled
export WANDB_ENTITY="draftrec"                    # workspace name
export WANDB_PROJECT="Isaaclab_Franka_Pickup"     # project name
export WANDB_LOG_EVERY=10                         # wandb log interval (steps)

# ── Environment ──
export NUM_TRAIN_ENVS=10                           # training envs
export EVAL_NUM_ENVS=10                           # eval envs (parallel batched)
export MAX_EPISODE_STEPS=1000                     # steps per episode
export CSV_BASE_DIR="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081825_758"

# ── GR00T base policy ──
export GROOT_MODEL_PATH="/home/t-kinamkim/Repos/VLA_RL/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000"
export GROOT_EMBODIMENT_TAG="new_embodiment"
export GROOT_POLICY_DEVICE="cuda:0"
export LANGUAGE_OVERRIDE="pick up mushroom"

# ── Algorithm ──
export TOTAL_TIMESTEPS=200000                     # 4x longer training
export LEARNING_STARTS=10000                      # warmup steps (1 env, base policy)
export CRITIC_WARMUP_STEPS=5000                    # critic-only updates
export BATCH_SIZE=256
export BUFFER_SIZE=200000
export GAMMA=0.99
export N_STEP=3
export OFFLINE_FRACTION=0.5                       # 50% offline from dense_clipped data
export OFFLINE_DATA_DIR="/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_dense_clipped_3cm_vlm"
export SUCCESS_THRESHOLD=0.03                     # cube lift success threshold (meters)
export REWARD_TYPE=dense_clipped                  # sparse, dense, dense_clipped
export CUBE_PERTURB_RANGE=0.05                    # cube XY perturbation range (meters)
export CUBE_PERTURB_TABLE="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/configs/cube_perturb_table.json"
export RANDOM_ACTION_NOISE_SCALE=0.0              # warmup noise scale (0=pure base policy)
export RANDOM_CUBE_PERTURB=true                   # randomize cube XY(±5cm) + yaw(±10°) each episode
export RESUME_CHECKPOINT="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/pickMushroom_resid_v17_vlm/20260316_072751/checkpoints/agent_step35000.pt"

# ── Exploration noise ──
export STDDEV_MAX=0.05                            # exploration noise (start)
export STDDEV_MIN=0.05                            # exploration noise (end)

# ── Network / optimiser ──
export ACTOR_LR=3e-7                              # actor learning rate (conservative)
export CRITIC_LR=1e-4                             # (set in config, not CLI)
export ACTION_SCALE=0.1                            # residual action scale (normalized space)

# ── Evaluation ──
export EVAL_INTERVAL=5000                            # eval every N train steps
export EVAL_FIRST=true                            # eval at step 0?
export SAVE_VIDEO=true                            # save eval videos?
export EVAL_SAVE_VIDEO=true                       # save eval videos in async eval?
export DISABLE_EVAL=true                          # disable inline eval (use async eval process)

# ── Checkpointing / output ──
export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/outputs/${EXP_NAME}/${RUN_TIMESTAMP}"
export CHECKPOINT_INTERVAL=5000                     # save checkpoint interval

# ── Debug flags ──
export DEBUG_ZERO_RESIDUAL=false                  # true = residual forced to 0 (base policy eval)

# ── Docker container ──
export CONTAINER="isaaclab_resfit"
export DEVICE="cuda:0"
