#!/bin/bash
#SBATCH --job-name=mj_timing
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=01:00:00

set -euo pipefail
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${HOME}/.local/lib/gl:${LD_LIBRARY_PATH:-}"
export DS_BUILD_OPS=0
export PYTHONUNBUFFERED=1
export WANDB_API_KEY="$(cat /home/nas_main/kinamkim/Keys/wandb_api_key)"

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/mujoco_cube_perturb_table.json \
    --asymmetric_critic \
    --total_timesteps 1000 \
    --eval_interval 500 \
    --eval_num_episodes 2 \
    --no_warmup \
    --async_eval \
    --ema_alpha 0.9 \
    --wandb_project mujoco-franka-residual-td3 \
    --wandb_name timing_test \
    --output_dir outputs/timing_test
