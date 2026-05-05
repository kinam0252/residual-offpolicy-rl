#!/bin/bash
#SBATCH --job-name=mj_v60_dense_gamma995
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_v60_dense_gamma995_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_v60_dense_gamma995_%j.err

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export WANDB_API_KEY="$(cat /home/nas_main/kinamkim/Keys/wandb_api_key)"

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/mujoco_cube_perturb_table.json \
    --reward_type dense \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --async_eval \
    --asymmetric_critic \
    --wandb_project mujoco-franka-residual-td3 \
    --wandb_entity draftrec \
    --wandb_name v60_dense_gamma995 \
    --output_dir outputs/mujoco_td3/v60_dense_gamma995 \
    --offline_data_dir outputs/offline_data/dense_v1_20260404_224233 \
    --action_l2_reg 0.1 --action_scale 0.2 --max_episode_steps 500 --stddev_max 0.05 --stddev_min 0.05 --offline_fraction 0.5 --gamma 0.995
