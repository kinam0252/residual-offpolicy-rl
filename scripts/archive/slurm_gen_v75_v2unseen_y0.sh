#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-extra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --job-name=gen_v75_v2unseen_y0
#SBATCH --output=/home/nas_main/kinamkim/slurms/gen_v75_v2unseen_y0_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/gen_v75_v2unseen_y0_%j.err

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:$LD_LIBRARY_PATH
export DS_BUILD_OPS=0
export PYTHONUNBUFFERED=1

python -m resfit.rl_finetuning.scripts.eval_checkpoint \
    --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --perturb_table configs/generalize_v2_unseen.json \
    --checkpoint outputs/mujoco_td3/v75_off02_yaw10/checkpoints/best.pt \
    --eval_num_episodes 2 \
    --asymmetric_critic \
    --action_scale 0.2 --actor_hidden_dim 512 --critic_hidden_dim 1024 \
    --seed 42 \
    --output_json outputs/eval_results/gen_v75_v2unseen_y0.json \
    --label gen_v75_v2unseen_y0
