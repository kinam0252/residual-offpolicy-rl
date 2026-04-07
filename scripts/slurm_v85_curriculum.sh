#!/bin/bash
#SBATCH --job-name=mj_v85_curric
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_v85_curric_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_v85_curric_%j.err

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

EASY='{"dx":[-3,11],"dy":[-10,10],"yaw":[0,0]}'
S0='{"step":0,"range":{"dx":[-3,11],"dy":[-10,10],"yaw":[0,0]}}'
S1='{"step":10000,"range":{"dx":[-5,12],"dy":[-12,12],"yaw":[-5,5]}}'
S2='{"step":20000,"range":{"dx":[-8,14],"dy":[-15,15],"yaw":[-10,10]}}'
CURRIC="[$S0,$S1,$S2]"

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --num_envs 10 \
    --random_cube_range "$EASY" \
    --curriculum_stages "$CURRIC" \
    --reward_type dense \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --async_eval \
    --asymmetric_critic \
    --wandb_project mujoco-franka-residual-td3 \
    --wandb_entity draftrec \
    --wandb_name v85_curriculum \
    --output_dir outputs/mujoco_td3/v85_curriculum \
    --offline_data_dir outputs/offline_data/dense_v2 \
    --action_l2_reg 0.1 --action_scale 0.2 --max_episode_steps 500 \
    --stddev_max 0.05 --stddev_min 0.05 --critic_warmup_steps 2000 \
    --offline_pretrain_only --offline_fraction 0.5
