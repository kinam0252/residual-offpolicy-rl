#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --job-name=as_sweep
#SBATCH --output=/home/nas_main/kinamkim/slurms/as_sweep_%A_%a.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/as_sweep_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --comment=pytorch

# ──────────────────────────────────────────────────────────────
# ActionScaler sweep for Cube Lift (66ep, hard)
#
# Based on best config: hard_v2_s42 (100% SR at step 8k)
# Now uses --use_action_scaler instead of physical scales.
#
# Usage (single variant):
#   VARIANT=v1 sbatch scripts/slurm_as_sweep_lift.sh
#
# Usage (all variants):
#   for v in v1 v2 v3 v4 v5; do VARIANT=$v sbatch scripts/slurm_as_sweep_lift.sh; done
# ──────────────────────────────────────────────────────────────

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export CUDA_HOME=~/fake_cuda
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CUDA 13.0 환경: PyTorch bundled NVIDIA libs + EGL
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/lib-compat:$NV/cu13/lib:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:~/.local/lib/gl:${LD_LIBRARY_PATH:-}

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

VARIANT=${VARIANT:-v1}

case $VARIANT in
    v1)
        # action_scale=0.1 → residual = 10% of data range
        # pos_x: ~10mm, pos_y: ~26mm, grip: ~0.025
        ACTION_SCALE=0.1
        L2_REG=0.01
        DESC="as01_l2001"
        ;;
    v2)
        # action_scale=0.3 → residual = 30% of data range
        # pos_x: ~30mm, pos_y: ~77mm, grip: ~0.075
        ACTION_SCALE=0.3
        L2_REG=0.01
        DESC="as03_l2001"
        ;;
    v3)
        # action_scale=0.5 → residual = 50% of data range
        ACTION_SCALE=0.5
        L2_REG=0.01
        DESC="as05_l2001"
        ;;
    v4)
        # action_scale=0.1 + no L2 (compare L2 effect)
        ACTION_SCALE=0.1
        L2_REG=0.0
        DESC="as01_noL2"
        ;;
    v5)
        # action_scale=1.0 → same as old physical scale=1.0 baseline equivalent
        ACTION_SCALE=1.0
        L2_REG=0.01
        DESC="as10_l2001"
        ;;
    *)
        echo "Unknown VARIANT=$VARIANT (use v1..v5)"
        exit 1
        ;;
esac

OUTPUT_DIR="outputs/mujoco_td3/gr00t_sim_66ep/as_sweep_hard_${DESC}"
WANDB_NAME="as_sweep_hard_${DESC}"

echo "=== ActionScaler Sweep ==="
echo "  VARIANT=$VARIANT"
echo "  action_scale=$ACTION_SCALE, l2_reg=$L2_REG"
echo "  output=$OUTPUT_DIR"

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_66ep/checkpoint-300000 \
    --reward_type dense_clipped \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --batch_size 256 \
    --buffer_size 500000 \
    --actor_lr 1e-5 --critic_lr 1e-4 \
    --gamma 0.99 \
    --target_tau 0.005 \
    --eval_perturb_table configs/eval_table_hard_v2.json \
    --random_cube_range '{"dx":[-12,16],"dy":[-20,20],"yaw":[0,0]}' \
    --num_envs 30 \
    --max_episode_steps 500 \
    --use_action_scaler \
    --action_scale ${ACTION_SCALE} \
    --random_action_noise_scale 0.05 \
    --action_l2_reg ${L2_REG} \
    --offline_data_dir /home/nas_main/kinamkim/Repos/Intern/outputs/offline_data/gr00t_sim_66ep/hard \
    --offline_fraction 0.5 \
    --critic_warmup_steps 2000 \
    --async_eval \
    --eval_num_episodes 2 \
    --asymmetric_critic \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 256 \
    --n_step 3 \
    --seed 42 \
    --use_calibrated_wrist \
    --output_dir ${OUTPUT_DIR} \
    --wandb_mode online \
    --wandb_project mujoco-resfit \
    --wandb_name ${WANDB_NAME}
