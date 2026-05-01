#!/bin/sh
#SBATCH --partition=sub
#SBATCH --qos=core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/eval_v60_s42_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/eval_v60_s42_%j.err
#SBATCH --job-name=eval_v60_s42

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:$LD_LIBRARY_PATH
export DS_BUILD_OPS=0
export PYTHONUNBUFFERED=1

python -m resfit.rl_finetuning.scripts.eval_checkpoint --checkpoint outputs/mujoco_td3/v60_dense_gamma995/checkpoints/best.pt --action_scale 0.2 --actor_hidden_dim 512 --critic_hidden_dim 1024 --perturb_table configs/mujoco_cube_perturb_table.json --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 --eval_num_episodes 2 --max_episode_steps 500 --reward_type dense --asymmetric_critic --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml --seed 42 --label v60_s42 --output_json outputs/eval_results/v60_s42.json
