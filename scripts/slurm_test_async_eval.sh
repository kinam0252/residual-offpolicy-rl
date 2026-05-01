#!/bin/bash
#SBATCH --job-name=mj_test_async
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=01:00:00

set -euo pipefail

# ── Environment ──
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.local/lib/gl:${LD_LIBRARY_PATH:-}"
export DS_BUILD_OPS=0
export PYTHONUNBUFFERED=1
export WANDB_API_KEY="$(cat /home/nas_main/kinamkim/Keys/wandb_api_key)"

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

echo "[$(date)] Starting test async eval on $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-?}"

python resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --perturb_table configs/mujoco_cube_perturb_table.json \
    --total_timesteps 200 \
    --eval_interval 50 \
    --eval_num_episodes 2 \
    --learning_starts 100 \
    --critic_warmup_steps 50 \
    --batch_size 64 \
    --no_warmup \
    --async_eval \
    --asymmetric_critic \
    --checkpoint_interval 100 \
    --wandb_project mujoco-franka-residual-td3 \
    --wandb_name test_async_eval \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --output_dir outputs/test_async_eval

echo "[$(date)] Done."
