#!/bin/bash
#SBATCH --job-name=mj_v96_70k_normal
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_v96_70k_normal_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_v96_70k_normal_%j.err
#SBATCH --time=72:00:00

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_sim_v2/checkpoint-70000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/perturb_table_sim_v2.json \
    --reward_type dense \
    --total_timesteps 500000 \
    --eval_interval 2000 \
    --checkpoint_interval 10000 \
    --async_eval \
    --asymmetric_critic \
    --wandb_project mujoco-franka-residual-td3 \
    --wandb_entity draftrec \
    --action_l2_reg 0.1 --action_scale 0.2 --max_episode_steps 500 \
    --critic_warmup_steps 2000 \
    --offline_data_dir outputs/offline_data/sim_groot_v2_70k_noema_normal \
    --offline_fraction 0.2 \
    --wandb_name v96_70k_normal \
    --output_dir outputs/mujoco_td3/v96_70k_normal \
    --stddev_max 0.05 --stddev_min 0.05 \
    --target_tau 0.005 \
    --random_cube_range '{"dx":[-8,14],"dy":[-15,15],"yaw":[-10,10]}'
